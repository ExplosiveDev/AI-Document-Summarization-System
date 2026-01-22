# AI Document Summarization System (Map-Reduce Architecture)
Stack: Python, PyTorch, Hugging Face Transformers, LangChain, PyMuPDF

Developed an abstractive summarization system for long-form PDF documents (books) based on BART/mT5 models and Transformer architecture.

Implemented a hierarchical recursive summarization algorithm (Map-Reduce), enabling the processing of unlimited text volume while maintaining contextual integrity.

Optimized inference stability and factual accuracy by fine-tuning stochastic decoding parameters, including Temperature (to control randomness) and Top-p (Nucleus Sampling) to filter low-probability tokens and reduce hallucinations.

Optimized VRAM usage by 70% through 4-bit quantization (BitsAndBytes), allowing LLM deployment on consumer-grade GPUs.

Integrated LangChain Text Splitters for intelligent text segmentation with overlapping chunks to minimize data loss at boundaries.

Built an interactive Gradio UI featuring real-time processing progress visualization.


<img width="900" height="450" alt="1" src="https://github.com/user-attachments/assets/afaf2e9a-3542-4d5f-8a4d-2bb92eb3f9f5" />
<img width="400" height="230" alt="2" src="https://github.com/user-attachments/assets/3ddaa536-d098-459f-adad-bd841e27a255" />
<img width="400" height="230" alt="3" src="https://github.com/user-attachments/assets/90d7f15f-4557-46e1-9024-486015dc13b2" />
<img width="400" height="230" alt="4" src="https://github.com/user-attachments/assets/3f01a2b1-6c6c-42c1-9358-1d263f9c5c6e" />
<img width="900" height="450" alt="5" src="https://github.com/user-attachments/assets/c8cff322-9fe4-4a40-aa3b-500d47d75634" />



<img width="1126" height="765" alt="Diagram" src="https://github.com/user-attachments/assets/f587a331-61cf-4e98-9587-e315de2d81f1" />
With this infrastructure, I can process any text with any size in English.


[Log_Robinson_Crusoe_BT.txt](https://github.com/user-attachments/files/24809352/Log_Robinson_Crusoe_BT.txt)

--- Processing level: 360 chunks ---

Robinson Crusoe was born in the year 1632, in the city of York, of a good family, though not of that country. His father was a foreigner of Bremen, who settled first at Hull and later at York. He had two elder brothers, one of whom was killed at the battle near Dunkirk against the famous Colonel Lockhart.
The author's father, a wise and grave man, gave him counsel against what he foresaw was his design. He told him it was men of desperate fortunes on one hand, or of aspiring, superior fortunes on the other, who were either too far above me or too far below me; that mine was the middle state, or what might be called the upper station of low life. He said kings have frequently lamented the miserable consequence of being born to great things, and wished they had been placed in the middle of the two extremes.
The middle station had the fewest disasters, and was not exposed to so many vicissitudes as the higher or lower part of mankind. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

--- Processing level: 72 chunks ---

Robinson Crusoe was born in the year 1632, in the city of York. His father was a foreigner of Bremen, who settled first at Hull and later at York. He had two elder brothers, one of whom was killed at the battle near Dunkirk. The author, a young man, wanted to go to sea and see the world. He decided that his father would not let him go if he stayed at home.
The author is the son of Robinson Crusoe, who was born in 1651. He fled to London on a ship bound for England after being told by his father that if he went abroad he would be 'the most miserable wretch that ever was born' The writer's mother refused to let him go, and his father warned her not to allow it to happen.
The story of Robinson Crusoe and his journey from Hull to Yarmouth is told in 487 pages. The author describes how he was driven on by an obstinacy that nothing could resist. He writes: ‘I had no power to go home, yet I had noPower to do it’
The author was a young man when he first set sail on a voyage to Guinea. He had been sent by his father, who wanted him to make money for himself. The captain of the ship taught him how to be a sailor and a merchant. He returned home with almost 300 pounds after his. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

--- Processing level: 15 chunks ---

The author is the son of Robinson Crusoe, who was born in 1651. He fled to London on a ship bound for England after being told by his father that if he went abroad he would be 'the most miserable wretch that ever was born' The writer's mother refused to let him go, and his father warned her not to allow it to happen.
The story of Robinson Crusoe is based on a novel by the same name written in 17th century England. The book was written by Henry VIII and published in 1837. It is about a young boy who goes to sea with his father to find water, but ends up being eaten by a wild beast. The author describes how he tried to kill the beast, but was unable to do so. He then decided to cut off the creature's head with a hatchet, which he did with great skill.
The story of Robinson Crusoe was written in the early part of the 19th century. The author describes how he tried to get back to shore with his raft after it ran aground on a shoal. He found himself on a barren island, uninhabited except by wild beasts. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

--- Processing level: 3 chunks ---

The story of Robinson Crusoe is based on a novel by the same name written in 17th century England. The book was written by Henry VIII and published in 1837. It is about a young boy who goes to sea with his father to find water, but ends up being eaten by a wild beast. The author describes how he tried to kill the beast, but was unable to do so. He then decided to cut off the creature's head with a hatchet, which he did with great skill.
The story of Robinson Crusoe was written in the first half of the 19th century. It is a tale of a man stranded on a ship, trying to find his way back to land. The story has been turned into a popular film, The Last Voyage, starring Tom Hanks.
The story of Robinson Crusoe is told in a series of novels, starting with The Hobbit. The novel was written by Henry James, and published in 1851. It is the first novel to be published in English since the publication of ‘The Hobbit’. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

--- Summarized text ---

"The story of Robinson Crusoe is based on a novel by the same name written in 17th century England. The book was written by Henry VIII and published in 1837. It is about a young boy who goes to sea with his father to find water, but ends up being eaten by a wild beast. The author describes how he tried to kill the beast, but was unable to do so. He then decided to cut off the creature's head with a hatchet, which he did with great skill.", 'The story of Robinson Crusoe was written in the first half of the 19th century. It is a tale of a man stranded on a ship, trying to find his way back to land. The story has been turned into a popular film, The Last Voyage, starring Tom Hanks.', 'The story of Robinson Crusoe is told in a series of novels, starting with The Hobbit. The novel was written by Henry James, and published in 1851. It is the first novel to be published in English since the publication of ‘The Hobbit’']
Keyboard interruption in main thread... closing server.
