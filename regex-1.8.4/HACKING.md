Your friendly guide to hacking and navigating the regex library.

This guide assumes familiarity with Rust and Cargo, and at least a perusal of
the user facing documentation for this crate.

If you're looking for background on the implementation in this library, then
you can do no better than Russ Cox's article series on implementing regular
expressions using finite automata: https://swtch.com/~rsc/regexp/


## Architecture overview

As you probably already know, this library executes regular expressions using
finite automata. In particular, a design goal is to make searching linear
with respect to both the regular expression and the text being searched.
Meeting that design goal on its own is not so hard and can be done with an
implementation of the Pike VM (similar to Thompson's construction, but supports
capturing groups), as described in: https://swtch.com/~rsc/regexp/regexp2.html
--- This library contains such an implementation in src/pikevm.rs.

Making it fast is harder. One of the key problems with the Pike VM is that it
can be in more than one state at any point in time, and must shuffle capture
positions between them. The Pike VM also spends a lot of time following the
same epsilon transitions over and over again. We can employ one trick to
speed up the Pike VM: extract one or more literal prefixes from the regular
expression and execute specialized code to quickly find matches of those
prefixes in the search text. The Pike VM can then be avoided for most the
search, and instead only executed when a prefix is found. The code to find
prefixes is in the regex-syntax crate (in this repository). The code to search
for literals is in src/literals.rs. When more than one literal prefix is found,
we fall back to an Aho-Corasick DFA using the aho-corasick crate. For one
literal, we use a variant of the Boyer-Moore algorithm. Both Aho-Corasick and
Boyer-Moore use `memchr` when appropriate. The Boyer-Moore variant in this
library also uses elementary frequency analysis to choose the right byte to run
`memchr` with.

Of course, detecting prefix literals can only take us so far. Not all regular
expressions have literal prefixes. To remedy this, we try another approach
to executing the Pike VM: backtracking, whose implementation can be found in
src/backtrack.rs. One reason why backtracking can be faster is that it avoids
excessive shuffling of capture groups. Of course, backtracking is susceptible
to exponential runtimes, so we keep track of every state we've visited to make
sure we never visit it again. This guarantees linear time execution, but we
pay for it with the memory required to track visited states. Because of the
memory requirement, we only use this engine on small search strings *and* small
regular expressions.

Lastly, the real workhorse of this library is the "lazy" DFA in src/dfa.rs.
It is distinct from the Pike VM in that the DFA is explicitly represented in
memory and is only ever in one state at a time. It is said to be "lazy" because
the DFA is computed as text is searched, where each byte in the search text
results in at most one new DFA state. It is made fast by caching states. DFAs
are susceptible to exponential state blow up (where the worst case is computing
a new state for every input byte, regardless of what's in the state cache). To
avoid using a lot of memory, the lazy DFA uses a bounded cache. Once the cache
is full, it is wiped and state computation starts over again. If the cache is
wiped too frequently, then the DFA gives up and searching falls back to one of
the aforementioned algorithms.

All of the above matching engines expose precisely the same matching semantics.
This is indeed tested. (See the section below about testing.)

The following sub-sections describe the rest of the library and how each of the
matching engines are actually used.

### Parsing

Regular expressions are parsed using the regex-syntax crate, which is
maintained in this repository. The regex-syntax crate defines an abstract
syntax and provides very detailed error messages when a parse error is
encountered. Parsing is done in a separate crate so that others may benefit
from its existence, and because it is relatively divorced from the rest of the
regex library.

The regex-syntax crate also provides sophisticated support for extracting
prefix and suffix literals from regular expressions.

### Compilation

The compiler is in src/compile.rs. The input to the compiler is some abstract
syntax for a regular expression and the output is a sequence of opcodes that
matching engines use to execute a search. (One can think of matching engines as
mini virtual machines.) The sequence of opcodes is a particular encoding of a
non-deterministic finite automaton. In particular, the opcodes explicitly rely
on epsilon transitions.

Consider a simple regular expression like `a|b`. Its compiled form looks like
this:

    000 Save(0)
    001 Split(2, 3)
    002 'a' (goto: 4)
    003 'b'
    004 Save(1)
    005 Match

The first column is the instruction pointer and the second column is the
instruction. Save instructions indicate that the current position in the input
should be stored in a captured location. Split instructions represent a binary
branch in the program (i.e., epsilon transitions). The instructions `'a'` and
`'b'` indicate that the literal bytes `'a'` or `'b'` should match.

In older versions of this library, the compilation looked like this:

    000 Save(0)
    001 Split(2, 3)
    002 'a'
    003 Jump(5)
    004 'b'
    005 Save(1)
    006 Match

In particular, empty instructions that merely served to move execution from one
point in the program to another were removed. Instead, every instruction has a
`goto` pointer embedded into it. This resulted in a small performance boost for
the Pike VM, because it was one fewer epsilon transition that it had to follow.

There exist more instructions and they are defined and documented in
src/prog.rs.

Compilation has several knobs and a few unfortunately complicated invariants.
Namely, the output of compilation can be one of two types of programs: a
program that executes on Unicode scalar values or a program that executes
on raw bytes. In the former case, the matching engine is responsible for
performing UTF-8 decoding and executing instructions using Unicode codepoints.
In the latter case, the program handles UTF-8 decoding implicitly, so that the
matching engine can execute on raw bytes. All matching engines can execute
either Unicode or byte based programs except for the lazy DFA, which requires
byte based programs. In general, both representations were kept because (1) the
lazy DFA requires byte based programs so that states can be encoded in a memory
efficient manner and (2) the Pike VM benefits greatly from inlining Unicode
character classes into fewer instructions as it results in fewer epsilon
transitions.

N.B. UTF-8 decoding is built into the compiled program by making use of the
utf8-ranges crate. The compiler in this library factors out common suffixes to
reduce the size of huge character classes (e.g., `\pL`).

A regrettable consequence of this split in instruction sets is we generally
need to compile two programs; one for NFA execution and one for the lazy DFA.

In fact, it is worse than that: the lazy DFA is not capable of finding the
starting location of a match in a single scan, and must instead execute a
backwards search after finding the end location. To execute a backwards search,
we must have compiled the regular expression *in reverse*.

This means that every compilation of a regular expression generally results in
three distinct programs. It would be possible to lazily compile the Unicode
program, since it is never needed if (1) the regular expression uses no word
boundary assertions and (2) the caller never asks for sub-capture locations.

### Execution

At the time of writing, there are four matching engines in this library:

1. The Pike VM (supports captures).
2. Bounded backtracking (supports captures).
3. Literal substring or multi-substring search.
4. Lazy DFA (no support for Unicode word boundary assertions).

Only the first two matching engines are capable of executing every regular
expression program. They also happen to be the slowest, which means we need
some logic that (1) knows various facts about the regular expression and (2)
knows what the caller wants. Using this information, we can determine which
engine (or engines) to use.

The logic for choosing which engine to execute is in src/exec.rs and is
documented on the Exec type. Exec values contain regular expression Programs
(defined in src/prog.rs), which contain all the necessary tidbits for actually
executing a regular expression on search text.

For the most part, the execution logic is straight-forward and follows the
limitations of each engine described above pretty faithfully. The hairiest
part of src/exec.rs by far is the execution of the lazy DFA, since it requires
a forwards and backwards search, and then falls back to either the Pike VM or
backtracking if the caller requested capture locations.

The Exec type also contains mutable scratch space for each type of matching
engine. This scratch space is used during search (for example, for the lazy
DFA, it contains compiled states that are reused on subsequent searches).

### Programs

A regular expression program is essentially a sequence of opcodes produced by
the compiler plus various facts about the regular expression (such as whether
it is anchored, its capture names, etc.).

### The regex! macro

The `regex!` macro no longer exists. It was developed in a bygone era as a
compiler plugin during the infancy of the regex crate. Back then, then only
matching engine in the crate was the Pike VM. The `regex!` macro was, itself,
also a Pike VM. The only advantages it offered over the dynamic Pike VM that
was built at runtime were the following:

  1. Syntax checking was done at compile time. Your Rust program wouldn't
     compile if your regex didn't compile.
  2. Reduction of overhead that was proportional to the size of the regex.
     For the most part, this overhead consisted of heap allocation, which
     was nearly eliminated in the compiler plugin.

The main takeaway here is that the compiler plugin was a marginally faster
version of a slow regex engine. As the regex crate evolved, it grew other regex
engines (DFA, bounded backtracker) and sophisticated literal optimizations.
The regex macro didn't keep pace, and it therefore became (dramatically) slower
than the dynamic engines. The only reason left to use it was for the compile
time guarantee that your regex is correct. Fortunately, Clippy (the Rust lint
tool) has a lint that checks your regular expression validity, which mostly
replaces that use case.

Additionally, the regex compiler plugin stopped receiving maintenance. Nobody
complained. At that point, it seemed prudent to just remove it.

Will a compiler plugin be brought back? The future is murky, but there is
definitely an opportunity there to build something that is faster than the
dynamic engines in some cases. But it will be challenging! As of now, there
are no plans to work on this.


## Testing

A key aspect of any mature regex library is its test suite. A subset of the
tests in this library come from Glenn Fowler's AT&T test suite (its online
presence seems gone at the time of writing). The source of the test suite is
located in src/testdata. The scripts/regex-match-tests.py takes the test suite
in src/testdata and generates tests/matches.rs.

There are also many other manually crafted tests and regression tests in
tests/tests.rs. Some of these tests were taken from RE2.

The biggest source of complexity in the tests is related to answering this
question: how can we reuse the tests to check all of our matching engines? One
approach would have been to encode every test into some kind of format (like
the AT&T test suite) and code generate tests for each matching engine. The
approach we use in this library is to create a Cargo.toml entry point for each
matching engine we want to test. The entry points are:

* `tests/test_default.rs` - tests `Regex::new`
* `tests/test_default_bytes.rs` - tests `bytes::Regex::new`
* `tests/test_nfa.rs` - tests `Regex::new`, forced to use the NFA
  algorithm on every regex.
* `tests/test_nfa_bytes.rs` - tests `Regex::new`, forced to use the NFA
  algorithm on every regex and use *arbitrary* byte based programs.
* `tests/test_nfa_utf8bytes.rs` - tests `Regex::new`, forced to use the NFA
  algorithm on every regex and use *UTF-8* byte based programs.
* `tests/test_backtrack.rs` - tests `Regex::new`, forced to use
  backtracking on every regex.
* `tests/test_backtrack_bytes.rs` - tests `Regex::new`, forced to use
  backtracking on every regex and use *arbitrary* byte based programs.
* `tests/test_backtrack_utf8bytes.rs` - tests `Regex::new`, forced to use
  backtracking on every regex and use *UTF-8* byte based programs.
* `tests/test_crates_regex.rs` - tests to make sure that all of the
  backends behave in the same way against a number of quickcheck
  generated random inputs. These tests need to be enabled through
  the `RUST_REGEX_RANDOM_TEST` environment variable (see
  below).

The lazy DFA and pure literal engines are absent from this list because
they cannot be used on every regular expression. Instead, we rely on
`tests/test_dynamic.rs` to test the lazy DFA and literal engines when possible.

Since the tests are repeated several times, and because `cargo test` runs all
entry points, it can take a while to compile everything. To reduce compile
times slightly, try using `cargo test --test default`, which will only use the
`tests/test_default.rs` entry point.

The random testing takes quite a while, so it is not enabled by default.
In order to run the random testing you can set the
`RUST_REGEX_RANDOM_TEST` environment variable to anything before
invoking `cargo test`. Note that this variable is inspected at compile
time, so if the tests don't seem to be running, you may need to run
`cargo clean`.

## Benchmarking

The benchmarking in this crate is made up of many micro-benchmarks. Currently,
there are two primary sets of benchmarks: the benchmarks that were adopted
at this library's inception (in `bench/src/misc.rs`) and a newer set of
benchmarks meant to test various optimizations. Specifically, the latter set
contain some analysis and are in `bench/src/sherlock.rs`. Also, the latter
set are all executed on the same lengthy input whereas the former benchmarks
are executed on strings of varying length.

There is also a smattering of benchmarks for parsing and compilation.

Benchmarks are in a separate crate so that its dependencies can be managed
separately from the main regex crate.

Benchmarking follows a similarly wonky setup as tests. There are multiple entry
points:

* `bench_rust.rs` - benchmarks `Regex::new`
* `bench_rust_bytes.rs` benchmarks `bytes::Regex::new`
* `bench_pcre.rs` - benchmarks PCRE
* `bench_onig.rs` - benchmarks Oniguruma

The PCRE and Oniguruma benchmarks exist as a comparison point to a mature
regular expression library. In general, this regex library compares favorably
(there are even a few benchmarks that PCRE simply runs too slowly on or
outright can't execute at all). I would love to add other regular expression
library benchmarks (especially RE2).

If you're hacking on one of the matching engines and just want to see
benchmarks, then all you need to run is:

    $ (cd bench && ./run rust)

If you want to compare your results with older benchmarks, then try:

    $ (cd bench && ./run rust | tee old)
    $ ... make it faster
    $ (cd bench && ./run rust | tee new)
    $ cargo benchcmp old new --improvements

The `cargo-benchcmp` utility is available here:
https://github.com/BurntSushi/cargo-benchcmp

The `./bench/run` utility can run benchmarks for PCRE and Oniguruma too. See
`./bench/bench --help`.

## Dev Docs

When digging your teeth into the codebase for the first time, the
crate documentation can be a great resource. By default `rustdoc`
will strip out all documentation of private crate members in an
effort to help consumers of the crate focus on the *interface*
without having to concern themselves with the *implementation*.
Normally this is a great thing, but if you want to start hacking
on regex internals it is not what you want. Many of the private members
of this crate are well documented with rustdoc style comments, and
it would be a shame to miss out on the opportunity that presents.
You can generate the private docs with:

```
$ rustdoc --crate-name docs src/lib.rs -o target/doc -L target/debug/deps --no-defaults --passes collapse-docs --passes unindent-comments
```

Then just point your browser at `target/doc/regex/index.html`.

See https://github.com/rust-lang/rust/issues/15347 for more info
about generating developer docs for internal use.
