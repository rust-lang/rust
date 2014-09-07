- Start Date: 2014-04-12
- RFC PR: [rust-lang/rfcs#42](https://github.com/rust-lang/rfcs/pull/42)
- Rust Issue: [rust-lang/rust#13700](https://github.com/rust-lang/rust/issues/13700)

# Summary

Add a `regexp` crate to the Rust distribution in addition to a small
`regexp_macros` crate that provides a syntax extension for compiling regular
expressions during the compilation of a Rust program.

The implementation that supports this RFC is ready to receive
feedback: https://github.com/BurntSushi/regexp

Documentation for the crate can be seen here:
http://burntsushi.net/rustdoc/regexp/index.html

regex-dna benchmark (vs. Go, Python):
https://github.com/BurntSushi/regexp/tree/master/benchmark/regex-dna

Other benchmarks (vs. Go):
https://github.com/BurntSushi/regexp/tree/master/benchmark

(Perhaps the links should be removed if the RFC is accepted, since I can't
guarantee they will always exist.)

# Motivation

Regular expressions provide a succinct method of matching patterns against
search text and are frequently used. For example, many programming languages
include some kind of support for regular expressions in its standard library.

The outcome of this RFC is to include a regular expression library in the Rust
distribution and resolve issue
[#3591](https://github.com/mozilla/rust/issues/3591).

# Detailed design

(Note: This is describing an existing design that has been implemented. I have
no idea how much of this is appropriate for an RFC.)

The first choice that most regular expression libraries make is whether or not
to include backreferences in the supported syntax, as this heavily influences
the implementation and the performance characteristics of matching text.

In this RFC, I am proposing a library that closely models Russ Cox's RE2
(either its C++ or Go variants). This means that features like backreferences
or generalized zero-width assertions are not supported. In return, we get
`O(mn)` worst case performance (with `m` being the size of the search text and
`n` being the number of instructions in the compiled expression).

My implementation currently simulates an NFA using something resembling the
Pike VM. Future work could possibly include adding a DFA. (N.B. RE2/C++
includes both an NFA and a DFA, but RE2/Go only implements an NFA.)

The primary reason why I chose RE2 was that it seemed to be a popular choice in
issue [#3591](https://github.com/mozilla/rust/issues/3591), and its worst case
performance characteristics seemed appealing. I was also drawn to the limited
set of syntax supported by RE2 in comparison to other regexp flavors.

With that out of the way, there are other things that inform the design of a
regexp library.

## Unicode

Given the already existing support for Unicode in Rust, this is a no-brainer.
Unicode literals should be allowed in expressions and Unicode character classes
should be included (e.g., general categories and scripts).

Case folding is also important for case insensitive matching. Currently, this
is implemented by converting characters to their uppercase forms and then
comparing them. Future work includes applying at least a simple fold, since
folding one Unicode character can produce multiple characters.

Normalization is another thing to consider, but like most other regexp
libraries, the one I'm proposing here does not do any normalization. (It seems
the recommended practice is to do normalization before matching if it's
needed.)

A nice implementation strategy to support Unicode is to implement a VM that
matches characters instead of bytes. Indeed, my implementation does this.
However, the public API of a regular expression library should expose *byte
indices* corresponding to match locations (which ought to be guaranteed to be
UTF8 codepoint boundaries by construction of the VM). My reason for this is
that byte indices result in a lower cost abstraction. If character indices are
desired, then a mapping can be maintained by the client at their discretion.

Additionally, this makes it consistent with the `std::str` API, which also
exposes byte indices.

## Word boundaries, word characters and Unicode

At least Python and D define word characters, word boundaries and space
characters with Unicode character classes. My implementation does the same
by augmenting the standard Perl character classes `\d`, `\s` and `\w` with
corresponding Unicode categories.

## Leftmost-first

As of now, my implementation finds the leftmost-first match. This is consistent
with PCRE style regular expressions.

I've pretty much ignored POSIX, but I think it's very possible to add
leftmost-longest semantics to the existing VM. (RE2 supports this as a
parameter, but I believe still does not fully comply with POSIX with respect to
picking the correct submatches.)

## Public API

There are three main questions that can be asked when searching text:

1. Does the string match this expression?
2. If so, where?
3. Where are its submatches?

In principle, an API could provide a function to only answer (3). The answers
to (1) and (2) would immediately follow. However, keeping track of submatches
is expensive, so it is useful to implement an optimization that doesn't keep
track of them if it doesn't have to. For example, submatches do not need to be
tracked to answer questions (1) and (2).

The rabbit hole continues: answering (1) can be more efficient than answering
(2) because you don't have to keep track of *any* capture groups ((2) requires
tracking the position of the full match). More importantly, (1) enables early
exit from the VM. As soon as a match is found, the VM can quit instead of
continuing to search for greedy expressions.

Therefore, it's worth it to segregate these operations. The performance
difference can get even bigger if a DFA were implemented (which can answer (1)
and (2) quickly and even help with (3)). Moreover, most other regular
expression libraries provide separate facilities for answering these questions
separately.

Some libraries (like Python's `re` and RE2/C++) distinguish between matching an
expression against an entire string and matching an expression against part of
the string. My implementation favors simplicity: matching the entirety of a
string requires using the `^` and/or `$` anchors. In all cases, an implicit
`.*?` is added the beginning and end of each expression evaluated. (Which is
optimized out in the presence of anchors.)

Finally, most regexp libraries provide facilities for splitting and replacing
text, usually making capture group names available with some sort of `$var`
syntax. My implementation provides this too. (These are a perfect fit for
Rust's iterators.)

This basically makes up the entirety of the public API, in addition to perhaps
a `quote` function that escapes a string so that it may be used as a literal in
an expression.

## The `regexp!` macro

With syntax extensions, it's possible to write an `regexp!` macro that compiles
an expression when a Rust program is compiled. This includes translating the
matching algorithm to Rust code specific to the expression given. This "ahead
of time" compiling results in a performance increase. Namely, it elides all
heap allocation.

I've called these "native" regexps, whereas expressions compiled at runtime are
"dynamic" regexps. The public API need not impose this distinction on users,
other than requiring the use of a syntax extension to construct a native
regexp. For example:

    let re = regexp!("a*");

After construction, `re` is indistinguishable from an expression created
dynamically:

    let re = Regexp::new("a*").unwrap();

In particular, both have the same type. This is accomplished with a
representation resembling:

    enum MaybeNative {
        Dynamic(~[Inst]),
        Native(fn(MatchKind, &str, uint, uint) -> ~[Option<uint>]),
    }

This syntax extension requires a second crate, `regexp_macros`, where the
`regexp!` macro is defined. Technically, this could be provided in the `regexp`
crate, but this would introduce a runtime dependency on `libsyntax` for any use
of the `regexp` crate.

[@alexcrichton
remarks](https://github.com/rust-lang/rfcs/pull/42#issuecomment-40320112)
that this state of affairs is a wart that will be corrected in the future.

## Untrusted input

Given worst case `O(mn)` time complexity, I don't think it's worth worrying
about unsafe search text.

Untrusted regular expressions are another matter. For example, it's very easy
to exhaust a system's resources with nested counted repetitions. For example,
`((a{100}){100}){100}` tries to create `100^3` instructions. My current
implementation does nothing to mitigate against this, but I think a simple hard
limit on the number of instructions allowed would work fine. (Should it be
configurable?)

## Name

The name of the crate being proposed is `regexp` and the type describing a
compiled regular expression is `Regexp`. I think an equally good name would be
`regex` (and `Regex`). Either name seems to be frequently used, e.g., "regexes"
or "regexps" in colloquial use. I chose `regexp` over `regex` because it
matches the name used for the corresponding package in Go's standard library.

Other possible names are `regexpr` (and `Regexpr`) or something with
underscores: `reg_exp` (and `RegExp`). However, I perceive these to be more
ugly and less commonly used than either `regexp` or `regex`.

Finally, we could use `re` (like Python), but I think the name could be
ambiguous since it's so short. `regexp` (or `regex`) unequivocally identifies
the crate as providing regular expressions.

For consistency's sake, I propose that the syntax extension provided be named
the same as the crate. So in this case, `regexp!`.

## Summary

My implementation is pretty much a port of most of RE2. The syntax should be
identical or almost identical. I think matching an existing (and popular)
library has benefits, since it will make it easier for people to pick it up and
start using it. There will also be (hopefully) fewer surprises. There is also
plenty of room for performance improvement by implementing a DFA.

# Alternatives

I think the single biggest alternative is to provide a backtracking
implementation that supports backreferences and generalized zero-width
assertions. I don't think my implementation precludes this possibility. For
example, a backtracking approach could be implemented and used only when
features like backreferences are invoked in the expression. However, this gives
up the blanket guarantee of worst case `O(mn)` time. I don't think I have the
wisdom required to voice a strong opinion on whether this is a worthwhile
endeavor.

Another alternative is using a binding to an existing regexp library. I think
this was discussed in issue
[#3591](https://github.com/mozilla/rust/issues/3591) and it seems like people
favor a native Rust implementation if it's to be included in the Rust
distribution. (Does the `regexp!` macro require it? If so, that's a huge
advantage.) Also, a native implementation makes it maximally portable.

Finally, it is always possible to persist without a regexp library.

# Unresolved questions

The public API design is fairly simple and straight-forward with no
surprises.  I think most of the unresolved stuff is how the backend is
implemented, which should be changeable without changing the public API (sans
adding features to the syntax).

I can't remember where I read it, but someone had mentioned defining a *trait*
that declared the API of a regexp engine. That way, anyone could write their
own backend and use the `regexp` interface. My initial thoughts are
YAGNI---since requiring different backends seems like a super specialized
case---but I'm just hazarding a guess here. (If we go this route, then we
might want to expose the regexp parser and AST and possibly the
compiler and instruction set to make writing your own backend easier. That
sounds restrictive with respect to making performance improvements in the
future.)

I personally think there's great value in keeping the standard regexp
implementation small, simple and fast. People who have more specialized needs
can always pick one of the existing C or C++ libraries.

For now, we could mark the API as `#[unstable]` or `#[experimental]`.

# Future work

I think most of the future work for this crate is to increase the performance,
either by implementing different matching algorithms (e.g., a DFA) or by
improving the code generator that produces native regexps with `regexp!`.

If and when a DFA is implemented, care must be taken when creating a code
generator, as the size of the code required can grow rapidly.

Other future work (that is probably more important) includes more Unicode
support, specifically for simple case folding.

