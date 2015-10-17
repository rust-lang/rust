# 2nd Report - ESOF

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
3. To be a good language for creating highly concurrent and highly safe systems.

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
