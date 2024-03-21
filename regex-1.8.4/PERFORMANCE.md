Your friendly guide to understanding the performance characteristics of this
crate.

This guide assumes some familiarity with the public API of this crate, which
can be found here: https://docs.rs/regex

## Theory vs. Practice

One of the design goals of this crate is to provide worst case linear time
behavior with respect to the text searched using finite state automata. This
means that, *in theory*, the performance of this crate is much better than most
regex implementations, which typically use backtracking which has worst case
exponential time.

For example, try opening a Python interpreter and typing this:

    >>> import re
    >>> re.search('(a*)*c', 'a' * 30).span()

I'll wait.

At some point, you'll figure out that it won't terminate any time soon. ^C it.

The promise of this crate is that *this pathological behavior can't happen*.

With that said, just because we have protected ourselves against worst case
exponential behavior doesn't mean we are immune from large constant factors
or places where the current regex engine isn't quite optimal. This guide will
detail those cases and provide guidance on how to avoid them, among other
bits of general advice.

## Thou Shalt Not Compile Regular Expressions In A Loop

**Advice**: Use `lazy_static` to amortize the cost of `Regex` compilation.

Don't do it unless you really don't mind paying for it. Compiling a regular
expression in this crate is quite expensive. It is conceivable that it may get
faster some day, but I wouldn't hold out hope for, say, an order of magnitude
improvement. In particular, compilation can take any where from a few dozen
microseconds to a few dozen milliseconds. Yes, milliseconds. Unicode character
classes, in particular, have the largest impact on compilation performance. At
the time of writing, for example, `\pL{100}` takes around 44ms to compile. This
is because `\pL` corresponds to every letter in Unicode and compilation must
turn it into a proper automaton that decodes a subset of UTF-8 which
corresponds to those letters. Compilation also spends some cycles shrinking the
size of the automaton.

This means that in order to realize efficient regex matching, one must
*amortize the cost of compilation*. Trivially, if a call to `is_match` is
inside a loop, then make sure your call to `Regex::new` is *outside* that loop.

In many programming languages, regular expressions can be conveniently defined
and compiled in a global scope, and code can reach out and use them as if
they were global static variables. In Rust, there is really no concept of
life-before-main, and therefore, one cannot utter this:

    static MY_REGEX: Regex = Regex::new("...").unwrap();

Unfortunately, this would seem to imply that one must pass `Regex` objects
around to everywhere they are used, which can be especially painful depending
on how your program is structured. Thankfully, the
[`lazy_static`](https://crates.io/crates/lazy_static)
crate provides an answer that works well:

    use lazy_static::lazy_static;
    use regex::Regex;

    fn some_helper_function(text: &str) -> bool {
        lazy_static! {
            static ref MY_REGEX: Regex = Regex::new("...").unwrap();
        }
        MY_REGEX.is_match(text)
    }

In other words, the `lazy_static!` macro enables us to define a `Regex` *as if*
it were a global static value. What is actually happening under the covers is
that the code inside the macro (i.e., `Regex::new(...)`) is run on *first use*
of `MY_REGEX` via a `Deref` impl. The implementation is admittedly magical, but
it's self contained and everything works exactly as you expect. In particular,
`MY_REGEX` can be used from multiple threads without wrapping it in an `Arc` or
a `Mutex`. On that note...

## Using a regex from multiple threads

**Advice**: The performance impact from using a `Regex` from multiple threads
is likely negligible. If necessary, clone the `Regex` so that each thread gets
its own copy. Cloning a regex does not incur any additional memory overhead
than what would be used by using a `Regex` from multiple threads
simultaneously. *Its only cost is ergonomics.*

It is supported and encouraged to define your regexes using `lazy_static!` as
if they were global static values, and then use them to search text from
multiple threads simultaneously.

One might imagine that this is possible because a `Regex` represents a
*compiled* program, so that any allocation or mutation is already done, and is
therefore read-only. Unfortunately, this is not true. Each type of search
strategy in this crate requires some kind of mutable scratch space to use
*during search*. For example, when executing a DFA, its states are computed
lazily and reused on subsequent searches. Those states go into that mutable
scratch space.

The mutable scratch space is an implementation detail, and in general, its
mutation should not be observable from users of this crate. Therefore, it uses
interior mutability. This implies that `Regex` can either only be used from one
thread, or it must do some sort of synchronization. Either choice is
reasonable, but this crate chooses the latter, in particular because it is
ergonomic and makes use with `lazy_static!` straight forward.

Synchronization implies *some* amount of overhead. When a `Regex` is used from
a single thread, this overhead is negligible. When a `Regex` is used from
multiple threads simultaneously, it is possible for the overhead of
synchronization from contention to impact performance. The specific cases where
contention may happen is if you are calling any of these methods repeatedly
from multiple threads simultaneously:

* shortest_match
* is_match
* find
* captures

In particular, every invocation of one of these methods must synchronize with
other threads to retrieve its mutable scratch space before searching can start.
If, however, you are using one of these methods:

* find_iter
* captures_iter

Then you may not suffer from contention since the cost of synchronization is
amortized on *construction of the iterator*. That is, the mutable scratch space
is obtained when the iterator is created and retained throughout its lifetime.

## Only ask for what you need

**Advice**: Prefer in this order: `is_match`, `find`, `captures`.

There are three primary search methods on a `Regex`:

* is_match
* find
* captures

In general, these are ordered from fastest to slowest.

`is_match` is fastest because it doesn't actually need to find the start or the
end of the leftmost-first match. It can quit immediately after it knows there
is a match. For example, given the regex `a+` and the haystack, `aaaaa`, the
search will quit after examining the first byte.

In contrast, `find` must return both the start and end location of the
leftmost-first match. It can use the DFA matcher for this, but must run it
forwards once to find the end of the match *and then run it backwards* to find
the start of the match. The two scans and the cost of finding the real end of
the leftmost-first match make this more expensive than `is_match`.

`captures` is the most expensive of them all because it must do what `find`
does, and then run either the bounded backtracker or the Pike VM to fill in the
capture group locations. Both of these are simulations of an NFA, which must
spend a lot of time shuffling states around. The DFA limits the performance hit
somewhat by restricting the amount of text that must be searched via an NFA
simulation.

One other method not mentioned is `shortest_match`. This method has precisely
the same performance characteristics as `is_match`, except it will return the
end location of when it discovered a match. For example, given the regex `a+`
and the haystack `aaaaa`, `shortest_match` may return `1` as opposed to `5`,
the latter of which being the correct end location of the leftmost-first match.

## Literals in your regex may make it faster

**Advice**: Literals can reduce the work that the regex engine needs to do. Use
them if you can, especially as prefixes.

In particular, if your regex starts with a prefix literal, the prefix is
quickly searched before entering the (much slower) regex engine. For example,
given the regex `foo\w+`, the literal `foo` will be searched for using
Boyer-Moore. If there's no match, then no regex engine is ever used. Only when
there's a match is the regex engine invoked at the location of the match, which
effectively permits the regex engine to skip large portions of a haystack.
If a regex is comprised entirely of literals (possibly more than one), then
it's possible that the regex engine can be avoided entirely even when there's a
match.

When one literal is found, Boyer-Moore is used. When multiple literals are
found, then an optimized version of Aho-Corasick is used.

This optimization is in particular extended quite a bit in this crate. Here are
a few examples of regexes that get literal prefixes detected:

* `(foo|bar)` detects `foo` and `bar`
* `(a|b)c` detects `ac` and `bc`
* `[ab]foo[yz]` detects `afooy`, `afooz`, `bfooy` and `bfooz`
* `a?b` detects `a` and `b`
* `a*b` detects `a` and `b`
* `(ab){3,6}` detects `ababab`

Literals in anchored regexes can also be used for detecting non-matches very
quickly. For example, `^foo\w+` and `\w+foo$` may be able to detect a non-match
just by examining the first (or last) three bytes of the haystack.

## Unicode word boundaries may prevent the DFA from being used

**Advice**: In most cases, `\b` should work well. If not, use `(?-u:\b)`
instead of `\b` if you care about consistent performance more than correctness.

It's a sad state of the current implementation. At the moment, the DFA will try
to interpret Unicode word boundaries as if they were ASCII word boundaries.
If the DFA comes across any non-ASCII byte, it will quit and fall back to an
alternative matching engine that can handle Unicode word boundaries correctly.
The alternate matching engine is generally quite a bit slower (perhaps by an
order of magnitude). If necessary, this can be ameliorated in two ways.

The first way is to add some number of literal prefixes to your regular
expression. Even though the DFA may not be used, specialized routines will
still kick in to find prefix literals quickly, which limits how much work the
NFA simulation will need to do.

The second way is to give up on Unicode and use an ASCII word boundary instead.
One can use an ASCII word boundary by disabling Unicode support. That is,
instead of using `\b`, use `(?-u:\b)`.  Namely, given the regex `\b.+\b`, it
can be transformed into a regex that uses the DFA with `(?-u:\b).+(?-u:\b)`. It
is important to limit the scope of disabling the `u` flag, since it might lead
to a syntax error if the regex could match arbitrary bytes. For example, if one
wrote `(?-u)\b.+\b`, then a syntax error would be returned because `.` matches
any *byte* when the Unicode flag is disabled.

The second way isn't appreciably different than just using a Unicode word
boundary in the first place, since the DFA will speculatively interpret it as
an ASCII word boundary anyway. The key difference is that if an ASCII word
boundary is used explicitly, then the DFA won't quit in the presence of
non-ASCII UTF-8 bytes. This results in giving up correctness in exchange for
more consistent performance.

N.B. When using `bytes::Regex`, Unicode support is disabled by default, so one
can simply write `\b` to get an ASCII word boundary.

## Excessive counting can lead to exponential state blow up in the DFA

**Advice**: Don't write regexes that cause DFA state blow up if you care about
match performance.

Wait, didn't I say that this crate guards against exponential worst cases?
Well, it turns out that the process of converting an NFA to a DFA can lead to
an exponential blow up in the number of states. This crate specifically guards
against exponential blow up by doing two things:

1. The DFA is computed lazily. That is, a state in the DFA only exists in
   memory if it is visited. In particular, the lazy DFA guarantees that *at
   most* one state is created for every byte of input. This, on its own,
   guarantees linear time complexity.
2. Of course, creating a new state for *every* byte of input means that search
   will go incredibly slow because of very large constant factors. On top of
   that, creating a state for every byte in a large haystack could result in
   exorbitant memory usage. To ameliorate this, the DFA bounds the number of
   states it can store. Once it reaches its limit, it flushes its cache. This
   prevents reuse of states that it already computed. If the cache is flushed
   too frequently, then the DFA will give up and execution will fall back to
   one of the NFA simulations.

In effect, this crate will detect exponential state blow up and fall back to
a search routine with fixed memory requirements. This does, however, mean that
searching will be much slower than one might expect. Regexes that rely on
counting in particular are strong aggravators of this behavior. For example,
matching `[01]*1[01]{20}$` against a random sequence of `0`s and `1`s.

In the future, it may be possible to increase the bound that the DFA uses,
which would allow the caller to choose how much memory they're willing to
spend.

## Resist the temptation to "optimize" regexes

**Advice**: This ain't a backtracking engine.

An entire book was written on how to optimize Perl-style regular expressions.
Most of those techniques are not applicable for this library. For example,
there is no problem with using non-greedy matching or having lots of
alternations in your regex.
