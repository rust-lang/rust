% The Rustonomicon

#### The Dark Arts of Advanced and Unsafe Rust Programming

# NOTE: This is a draft document, and may contain serious errors

> Instead of the programs I had hoped for, there came only a shuddering blackness
and ineffable loneliness; and I saw at last a fearful truth which no one had
ever dared to breathe before — the unwhisperable secret of secrets — The fact
that this language of stone and stridor is not a sentient perpetuation of Rust
as London is of Old London and Paris of Old Paris, but that it is in fact
quite unsafe, its sprawling body imperfectly embalmed and infested with queer
animate things which have nothing to do with it as it was in compilation.

This book digs into all the awful details that are necessary to understand in
order to write correct Unsafe Rust programs. Due to the nature of this problem,
it may lead to unleashing untold horrors that shatter your psyche into a billion
infinitesimal fragments of despair.

Should you wish a long and happy career of writing Rust programs, you should
turn back now and forget you ever saw this book. It is not necessary. However
if you intend to write unsafe code -- or just want to dig into the guts of the
language -- this book contains invaluable information.

Unlike [The Book][trpl] we will be assuming considerable prior knowledge. In
particular, you should be comfortable with basic systems programming and Rust.
If you don't feel comfortable with these topics, you should consider [reading
The Book][trpl] first. Though we will not be assuming that you have, and will
take care to occasionally give a refresher on the basics where appropriate. You
can skip straight to this book if you want; just know that we won't be
explaining everything from the ground up.

To be clear, this book goes into deep detail. We're going to dig into
exception-safety, pointer aliasing, memory models, and even some type-theory.
We will also be spending a lot of time talking about the different kinds
of safety and guarantees.

[trpl]: ../book/index.html
