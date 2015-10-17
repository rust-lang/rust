# 2nd Report - ESOF

## Index

1. [Definition of Requirements](#definition-of-requirements)
2. [RUST's requirements](#rust's-requirements)
  1. [Bugs](#bugs)
3. [Pull Requests](#pull-requests)
4. [New Features](#new-features)
  1. [RFC](#rfc)
5. [Use Cases](#use-cases)

## RUST - Programming Language

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


#### Bugs

The Rust community welcomes anyone that reports Bugs. Even if the person that reports isn't really sure about what they are reporting.

It is asked to search in the existing issues for the bug before reporting it. However there is no problem in reporting a bug for the second time considering the enormous amount of existing issues.

All issues should fallow the form bellow:

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
   new tests or changes on old tests and the description of what the change is justifying and why it's necessary.

2. If it's a breaking change then it should include the string **[breaking-change]**.

3. The pull request will be assigned to a reviewer, though you can also choose a reviewer yourself.
	The reviewer will comment on your code, pointing out possible bugs, style issues, missing tests,
	or other problems.
	Once the reviewer thinks that the code is acceptable to be merged, they will sign off with an r+
	that indicates that it has passed review and is ready to be merged.

4.  Once the pull request is aproved, a bot called [Homu] that queues up the approved pull request and tries merging.


[Homu]:https://github.com/barosl/homu

### New features

To propose new features:
1. Discussion pre-RFC on the Rust Internals forum https://internals.rust-lang.org/, Reddit or Rust's personal blog
2. Do an RFC
	An RFC describes why you want a feature, a detailed description, possible drawbacks and possible alternatives.
3. Once the RFC is declared to be in final comment period the RFC will be accepted, postponed or rejected.
	Accepted: The RFC is merged into the master branch of the rfcs repo
	Postponed: The ticket is filed against the RFC repo to keep track on the discussion
	Rejected.

_NOTE: If anyone wants to propose a new feature to the Rust language, an issue should be opened in the [RFCs repository] instead of the Rust one. Then those new features will go through the RFC process._

[RFCs repository]:https://github.com/rust-lang/rfcs/issues/new

#### RFC

>The "RFC" (request for comments) process is intended to provide a consistent and controlled path for new features to enter the language and standard libraries, so that all stakeholders can be confident about the direction the language is evolving in.
>
>_From [Rust RFC repository]_
[Rust RFC repository]:https://github.com/rust-lang/rfcs/

### Use Cases

![alt tab](https://github.com/martapips/rust/blob/master/ESOF-docs/res/rustUseCaseDiagram.jpg?raw=true)
