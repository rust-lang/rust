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
* To cover the complete feature-set of C++, or any other language. It should provide majority-case features.
* To be 100% static, 100% safe, 100% reflective, or too dogmatic in any other sense. Trade-offs exist.
* To run on "every possible platform". It must eventually work without unnecessary compromises on widely-used hardware and software platforms.

# Is any part of this thing production-ready?

No. Feel free to play around, but don't expect completeness or stability yet. Expect incompleteness and breakage.

# Is this a completely Mozilla-planned and orchestrated thing?

No. It started as a Graydon Hoare's part-time side project in 2006 and remained so for over 3 years. Mozilla got involved in 2009 once the language was mature enough to run some basic tests and demonstrate the idea. Though it is sponsored by Mozilla, Rust is developed by a diverse community of enthusiasts.

# What will Mozilla use Rust for?

Mozilla intends to use Rust as a platform for prototyping experimental browser architectures. Specifically, the hope is to develop a browser that is more amenable to parallelization than existing ones, while also being less prone to common C++ coding errors that result in security exploits. The name of that project is _[Servo](http://github.com/mozilla/servo)_.

# Why a BSD-style permissive license rather than MPL or tri-license?

* Partly due to preference of the original developer (Graydon).
* Partly due to the fact that languages tend to have a wider audience and more diverse set of possible embeddings and end-uses than focused, coherent products such as web browsers. We'd like to appeal to as many of those potential contributors as possible.

# Why dual MIT/ASL2 license?

The Apache license includes important protection against patent aggression, but it is not compatible with the GPL, version 2. To avoid problems using Rust with GPL2, it is alternately MIT licensed.
