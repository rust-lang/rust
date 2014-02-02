% Project FAQ

# What is this project's goal, in one sentence?

To design and implement a safe, concurrent, practical, static systems language.

# Why are you doing this?

Existing languages at this level of abstraction and efficiency are unsatisfactory. In particular:

* Too little attention paid to safety.
* Poor concurrency support.
* Lack of practical affordances, too dogmatic about paradigm.

# What are some non-goals?

* To employ any particularly cutting-edge technologies. Old, established techniques are better.
* To prize expressiveness, minimalism or elegance above other goals. These are desirable but subordinate goals.
* To cover the "systems language" part all the way down to "writing an OS kernel".
* To cover the complete feature-set of C++, or any other language. It should provide majority-case features.
* To be 100% static, 100% safe, 100% reflective, or too dogmatic in any other sense. Trade-offs exist.
* To run on "every possible platform". It must eventually work without unnecessary compromises on widely-used hardware and software platforms.

# Is any part of this thing production-ready?

No. Feel free to play around, but don't expect completeness or stability yet. Expect incompleteness and breakage.

What exists presently is:

* A self-hosted (written in Rust) compiler, which uses LLVM as a backend.
* A runtime library.
* An evolving standard library.
* Documentation for the language and libraries.
* Incomplete tools for packaging and documentation.
* A test suite covering the compiler and libraries.

# Is this a completely Mozilla-planned and orchestrated thing?

No. It started as a part-time side project in 2006 and remained so for over 3 years. Mozilla got involved in 2009 once the language was mature enough to run some basic tests and demonstrate the idea.

# Why did you do so much work in private?

* A certain amount of shyness. Language work is somewhat political and flame-inducing.
* Languages designed by committee have a poor track record. Design coherence is important. There were a lot of details to work out and the initial developer (Graydon) had this full time job thing eating up most days.

# Why publish it now?

* The design is stable enough. All the major pieces have reached non-imaginary, initial implementation status. It seems to hold together ok.
* Languages solely implemented and supported by one person _also_ have a poor track record. To survive it'll need help.

# What will Mozilla use Rust for?

Mozilla intends to use Rust as a platform for prototyping experimental browser architectures. Specifically, the hope is to develop a browser that is more amenable to parallelization than existing ones, while also being less prone to common C++ coding errors. The name of that project is _[Servo](http://github.com/mozilla/servo)_.

# Are you going to use this to suddenly rewrite the browser and change everything? Is the Mozilla Corporation trying to force the community to use a new language?

No. This is a research project. The point is to explore ideas. There is no plan to incorporate any Rust-based technology into Firefox.

# Why GitHub rather than the normal Mozilla setup (Mercurial / Bugzilla / Tinderbox)?

* This is a fresh codebase and has no existing ties to Mozilla infrastructure; there is no particular advantage to (re)using any of the above infrastructure, it would all have to be upgraded and adapted to our needs.
* Git has been progressing rapidly in the years since Mozilla picked Mercurial for its main development needs, and appears to be both more widely known and more accessible at this point.
* This reduces the administrative requirements for contributing to merely establishing a paper trail via a contributor agreement. There is no need for vouching, granting commit access to Mozilla facilities, or setting up Mozilla user accounts.

# Why a BSD-style license rather than MPL or tri-license?

* Partly due to preference of the original developer (Graydon).
* Partly due to the fact that languages tend to have a wider audience and more diverse set of possible embeddings and end-uses than focused, coherent products such as web browsers. We'd like to appeal to as many of those potential contributors as possible.
