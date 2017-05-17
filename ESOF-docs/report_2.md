# 2nd Report - ESOF

## Index

1. [Definition of Requirements](#definition-of-requirements)
2. [RUST's requirements](#rust's-requirements)
  1. [Bugs](#bugs)
3. [Pull Requests](#pull-requests)
4. [New Features](#new-features)
  1. [RFC](#rfc)
5. [Use Cases](#use-cases)
6. [Critical Analysis] (#critical-analysis)

## RUST - Programming Language


Rust is a programming language focused on **stability**, **community** and **clarity**.
Stability is discussed quite a bit in their [blog] introducing a release channel and stabilization process.
Community has always been one of Rust's greatest strengths. They've introduced and refined the [RFC process], culminating with [subteams] to manage RFCs in each particular area. 
Community-wide debate on RFCs was indispensable for delivering a quality 1.0 release. 

All of this refinement prior to 1.0 was in service of reaching clarity on what Rust represents.

[blog]:http://blog.rust-lang.org/2014/10/30/Stability.html
[RFC process]:https://github.com/rust-lang/rfcs#when-you-need-to-follow-this-process
[subteams]:https://github.com/rust-lang/rfcs/pull/1068

### Definition of Requirements

> Software Requirements is a field within software engineering that deals with establishing the needs of stakeholders that are to be solved by software. The IEEE Standard Glossary of Software Engineering Technology defines a software requirement as:

>> 1. A condition or capability needed by a user to solve a problem or achieve an objective;

>> 2. A condition or capability that must be met or possessed by a system or system component to satisfy a contract, standard, specification, or other formally imposed document;

>> 3. A documented representation of a condition or capability as in 1 or 2.

>  _Requirement definition from [Wikipedia]_
[Wikipedia]: www.sapo.pt

As said above, requirements are conditions that the costumer wants the software to have before it is delivered.

### RUST's requirements

1. To be a fast systems programming language;
2. Guarantee memory safety;
3. To be a good language for creating highly concurrent and highly safe systems;
4. Zero-cost abstractions while still allowing precise control like a low-level language.

Rust is being developed by a huge community which counts with numerous ways to communicate such as the questions [Forum], [Twitter], [StackOverflow] or [Reddit].

[Forum]:https://internals.rust-lang.org/
[Twitter]:https://twitter.com/rustlang
[StackOverflow]:http://stackoverflow.com/questions/tagged/rust
[Reddit]:https://www.reddit.com/r/rust/

> Altogether, Rust is exciting because it is empowering: you can hack without fear.
And you can do so in contexts you might not have before, dropping down from languages like Ruby or Python, making your first foray into systems programming.

>  _From: [Rust in 2016]_
[Rust in 2016]: http://blog.rust-lang.org/2015/08/14/Next-year.html

#### Bugs

The Rust community welcomes anyone that reports Bugs. Even if the person that reports isn't really sure about what they are reporting.

It is asked to search in the existing issues for the bug before reporting it. However there is no problem in reporting a bug for the second time considering the enormous amount of existing issues.

All issues should follow the form bellow:

```
<short summary of the bug>

I tried this code:

<code sample that causes the bug>

I expected to see this happen: <explanation>

Instead, this happened: <explanation>

## Meta

`rustc --version --verbose`:

Backtrace:
```

>All three components are important: what you did, what you expected, what happened instead. Please include the output of rustc --version --verbose, which includes important information about what platform you're on, what version of Rust you're using, etc.
>
>  _From [Rust - Github]_
[Rust - Github]:https://github.com/rust-lang/rust

### Pull Requests

1.  The pull request that is commited on the rust repo should include the changes that the user made,
   new tests or changes on old tests and the description of what the change is, justifying and why it's necessary.

2. If it is a breaking change, which now that 1.0 has been released and there are stability guarantees in place should only happen to unstable features, then it should include the string **[breaking-change]** so these can easily be found and documented. These are collected in [bitrust] , to make it easy to find all of them.

3. The pull request will be assigned to a reviewer, though you can also choose a reviewer yourself.
	The reviewer will comment on your code, pointing out possible bugs, style issues, missing tests,
	or other problems.
	Once the reviewer thinks that the code is acceptable to be merged, they will sign off with an r+
	that indicates that it has passed review and is ready to be merged.

4.  Once the pull request is aproved, a bot called [Homu] that queues up the [approved pull request] and tries merging.
	[Here's] a simple example that demonstrates the code review and continuous integration process.

Certain types of pull requests, such as documentation pull requests, comment fixes, and the like, are very unlikely to fail. Doing full builds and tests can take a long time, so these pull requests are frequently manually collected into a "roll-up" branch, that can all be merged with one single merge and test of the full branch, rather than one for each pull request.

[Homu]:https://github.com/barosl/homu
[bitrust]:https://killercup.github.io/bitrust/
[approved pull request]:http://buildbot.rust-lang.org/homu/queue/rust
[Here's]:https://github.com/rust-lang/rust/pull/28729

### New features

To propose new features:

1. Discussion pre-RFC on [the Rust Internals forum], [Rust's subreddit] or [Rust's personal blog].

2. Do an RFC:
Features are proposed as part of a Request for Comments, generally referred to as an RFC. An RFC describes why you want a feature, a detailed description of the feature that is sufficient to implement it, possible drawbacks of the design, possible alternatives that could achieve the same goal, and any still open questions that are unresolved.

3. Once the RFC is declared to be in final comment period the RFC will be accepted, postponed or rejected.
	
	**Accepted**: The RFC is merged into the **[master]** branch of the **[rfcs]** repo.
	
	**Postponed**: The [ticket is filed against the RFC repo] to keep track on the discussion.
	
	**Rejected**: This means that the team believes that this feature is unlikely to ever get implemented.

4. If the RFC is accepted, a [tracking issue](with label B-RFC-approved) is filled against the rust repo to help keeping track of the implementation status of the RFC. 

When a feature is first implemented, it is marked as unstable, and you can only use it by explicitly opting in, and using the nightly compiler. This gives the community a chance to evaluate the feature, make sure it works as intended, ensure that it does not break existing code or cause problems that were unanticipated when merely discussing it. This process is described in the [Stability as a deliverable] blog post.
After there has been a certain amount of experience using the feature in the unstable form, along with any fixes that might be required, it is eventually promoted to being stable.

_NOTE: If anyone wants to propose a new feature to the Rust language, an issue should be opened in the [RFCs repository] instead of the Rust one. Then those new features will go through the RFC process._

[RFCs repository]:https://github.com/rust-lang/rfcs/issues/new
[the Rust Internals forum]:https://internals.rust-lang.org/
[Rust's subreddit]:https://www.reddit.com/r/rust/
[Rust's personal blog]:http://blog.rust-lang.org/
[ticket is filed against the RFC repo]:https://github.com/rust-lang/rfcs/issues?q=is%3Aissue+label%3Apostponed
[tracking issue]:https://github.com/rust-lang/rust/issues?q=is%3Aopen+is%3Aissue+label%3AB-RFC-approved
[Stability as a deliverable]:http://blog.rust-lang.org/2014/10/30/Stability.html
#### RFC

>The "RFC" (request for comments) process is intended to provide a consistent and controlled path for new features to enter the language and standard libraries, so that all stakeholders can be confident about the direction the language is evolving in.
>
>_From [Rust RFC repository]_
[Rust RFC repository]:https://github.com/rust-lang/rfcs/

### Use Cases

![alt tab](https://github.com/martapips/rust/blob/master/ESOF-docs/res/rustUseCaseDiagram.jpg?raw=true)

### Critical Analysis

In conclusion, we can say that Rust's team is focused on their requirements and it's developing them each day. 
The policy on adding new features is strict and well organized which allows Rust to be safer and more stable in the pursuit of the main goals.
The core team is well adjusted and is now working on some new improvements for the next year like doubling down on infrastructure, zeroing in on gaps in key features and
branching out into new places to use Rust.

The Rust community is very active and keeps growing, which leads to a better development of this programming language. 

We can see in their [personal blog] that the team is very commited since the beggining and it keeps updating Rust and bringing us news frequently, letting users know what is going on and allowing them to keep contributing.

[personal blog]:http://blog.rust-lang.org/

