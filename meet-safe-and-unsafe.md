% Meet Safe and Unsafe

Safe and Unsafe are Rust's chief engineers.

TODO: ADORABLE PICTURES OMG

Unsafe handles all the dangerous internal stuff. They build the foundations
and handle all the dangerous materials. By all accounts, Unsafe is really a bit
unproductive, because the nature of their work means that they have to spend a
lot of time checking and double-checking everything. What if there's an earthquake
on a leap year? Are we ready for that? Unsafe better be, because if they get
*anything* wrong, everything will blow up! What Unsafe brings to the table is
*quality*, not quantity. Still, nothing would ever get done if everything was
built to Unsafe's standards!

That's where Safe comes in. Safe has to handle *everything else*. Since Safe needs
to *get work done*, they've grown to be fairly careless and clumsy! Safe doesn't worry
about all the crazy eventualities that Unsafe does, because life is too short to deal
with leap-year-earthquakes. Of course, this means there's some jobs that Safe just
can't handle. Safe is all about quantity over quality.

Unsafe loves Safe to bits, but knows that they *can never trust them to do the
right thing*. Still, Unsafe acknowledges that not every problem needs quite the
attention to detail that they apply. Indeed, Unsafe would *love* if Safe could do
*everything* for them. To accomplish this, Unsafe spends most of their time
building *safe abstractions*. These abstractions handle all the nitty-gritty
details for Safe, and choose good defaults so that the simplest solution (which
Safe will inevitably use) is usually the *right* one. Once a safe abstraction is
built, Unsafe ideally needs to never work on it again, and Safe can blindly use
it in all their work.

Unsafe's attention to detail means that all the things that they mark as ok for
Safe to use can be combined in arbitrarily ridiculous ways, and all the rules
that Unsafe is forced to uphold will never be violated. If they *can* be violated
by Safe, that means *Unsafe*'s the one in the wrong. Safe can work carelessly,
knowing that if anything blows up, it's not *their* fault. Safe can also call in
Unsafe at any time if there's a hard problem they can't quite work out, or if they
can't meet the client's quality demands. Of course, Unsafe will beg and plead Safe
to try their latest safe abstraction first!

In addition to being adorable, Safe and Unsafe are what makes Rust possible.
Rust can be thought of as two different languages: Safe Rust, and Unsafe Rust.
Any time someone opines the guarantees of Rust, they are almost surely talking about
Safe. However Safe is not sufficient to write every program. For that,
we need the Unsafe superset.

Most fundamentally, writing bindings to other languages
(such as the C exposed by your operating system) is never going to be safe. Rust
can't control what other languages do to program execution! However Unsafe is
also necessary to construct fundamental abstractions where the type system is not
sufficient to automatically prove what you're doing is sound.

Indeed, the Rust standard library is implemented in Rust, and it makes substantial
use of Unsafe for implementing IO, memory allocation, collections,
synchronization, and other low-level computational primitives.

Upon hearing this, many wonder why they would not simply just use C or C++ in place of
Rust (or just use a "real" safe language). If we're going to do unsafe things, why not
lean on these much more established languages?

The most important difference between C++ and Rust is a matter of defaults:
Rust is 100% safe by default. Even when you *opt out* of safety in Rust, it is a modular
action. In deciding to work with unchecked uninitialized memory, this does not
suddenly make dangling or null pointers a problem. When using unchecked indexing on `x`,
one does not have to suddenly worry about indexing out of bounds on `y`.
C and C++, by contrast, have pervasive unsafety baked into the language. Even the
modern best practices like `unique_ptr` have various safety pitfalls.

It cannot be emphasized enough that Unsafe should be regarded as an exceptional
thing, not a normal one. Unsafe is often the domain of *fundamental libraries*: anything that needs
to make FFI bindings or define core abstractions. These fundamental libraries then expose
a safe interface for intermediate libraries and applications to build upon. And these
safe interfaces make an important promise: if your application segfaults, it's not your
fault. *They* have a bug.

And really, how is that different from *any* safe language? Python, Ruby, and Java libraries
can internally do all sorts of nasty things. The languages themselves are no
different. Safe languages *regularly* have bugs that cause critical vulnerabilities.
The fact that Rust is written with a healthy spoonful of Unsafe is no different.
However it *does* mean that Rust doesn't need to fall back to the pervasive unsafety of
C to do the nasty things that need to get done.

