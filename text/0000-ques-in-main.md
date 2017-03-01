- Feature Name: ques_in_main
- Start Date: 2017-02-22
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Allow the `?` operator to be used in `main`, and in `#[test]`
functions and doctests.

To make this possible, the return type of these functions are
generalized from `()` to a new trait, provisionally called
`Termination`.  libstd implements this trait for `!`, `()`, `Error`,
`Result<T, E> where T: Termination, E: Termination`, and possibly
other types TBD.  Applications can provide impls themselves if they
want.

There is no magic added to function signatures in rustc.  If you want
to use `?` in either `main` or a `#[test]` function you have to write
`-> Result<(), ErrorT>` (or whatever) yourself.  However, the rustdoc
template for doctests that are just a function body will be adjusted,
so that `?` can be used without having to write a function head yourself.

[Pre-RFC discussion][pre-rfc].  [Prior RFC issue][old-issue].

[pre-rfc]: https://internals.rust-lang.org/t/rfc-mentoring-opportunity-permit-in-main/4600
[old-issue]: https://github.com/rust-lang/rfcs/issues/1176

# Motivation
[motivation]: #motivation

It is currently not possible to use `?` in `main`, because `main`'s
return type is required to be `()`.  This is a trip hazard for new
users of the language, and complicates "programming in the small".

On a related note, `main` returning `()` means that short-lived
programs, designed to be invoked from the Unix shell or a similar
environment, have to contain extra boilerplate in order to comply with
those environments' conventions.  A typical construction is

``` rust
fn inner_main() -> Result<(), ErrorT> {
    // ... stuff which may fail ...
    Ok(())
}

fn main() -> () {
    use std::process::exit;
    use std::io::{Write,stderr};
    use libc::{EXIT_SUCCESS, EXIT_FAILURE};

    exit(match inner_main() {
        Ok(_) => EXIT_SUCCESS,

        Err(ref err) => {
            let progname = get_program_name();
            writeln!(stderr(), "{}: {}\n", progname, err);

            EXIT_FAILURE
        }
    })
}
```

Both of these problems can be solved at once if the compiler and/or
libstd are taught to recognize a `main` that returns
`Result<(), ErrorT>` and supply boilerplate similar to the above.

# Detailed design
[design]: #detailed-design

The design goals for this new feature are, in decreasing order of
importance:

1. The `?` operator should be usable in `main` (and `#[test]`
   functions and doctests).  This entails these functions now
   returning a richer value than `()`.
1. Existing code with `fn main() -> ()` should not break.
1. Errors returned from `main` in a hosted environment should
   *not* trigger a panic, consistent with the general language
   principle that panics are only for bugs.
1. We should take this opportunity to increase consistency with
   platform conventions for process termination. These often include
   the ability to pass an "exit status" up to some outer environment,
   conventions for what that status means, and an expectation that a
   diagnostic message will be generated when a program fails
   due to a system error.
1. We should avoid making life more complicated for people who don't
   care; on the other hand, if the Easiest Thing is also the Right
   Thing according to the platform convention, that is better all
   around.

Goal 1 dictates that the new return type for `main` will be
`Result<T,E>` for some T and E.  To minimize the necessary changes to
existing code that wants to start using `?` in `main`, T should be
_allowed_ to be `()`, but other types in that position may also make
sense.  The boilerplate shown [above][motivation] will work for any E
that is `Display`, but that is probably too general for the stdlib; we
propose only `Error` types should work by default.

[grep]: http://man7.org/linux/man-pages/man1/grep.1.html

## The `Termination` trait
[the-termination-trait]: #the-termination-trait

When `main` returns a nontrivial value, the runtime needs to know two
things about it: what error message, if any, to print, and what value
to pass to `std::process::exit`.  These are naturally encapsulated in
a trait, which we are tentatively calling `Termination`, with this
signature:

``` rust
trait Termination {
    fn write_diagnostic(&self, progname: &str, stream: &mut Write) -> ();
    fn exit_status(&self) -> i32;
}
```

`write_diagnostic` shall write a complete diagnostic message for
`self` to its `stream` argument.  If it produces no output, there will
be no output.  `stream` will be the same stream that `panic` messages
from the main thread would go to, which is normally "standard error".
(An application-controllable override mechanism may make sense,
[see below][squelching-diagnostics].)

The `progname` argument is a short version of the program's name
(abstractly, what you would type at the command line to start the
program, assuming it were in `PATH`; concretely, the basename of
`argv[0]`, with any trailing `.exe` or `.com` chopped off on Windows,
but not on other platforms).  This makes it easy to generate
diagnostics in the style conventional for "Unixy" systems, e.g.

    grep: /etc/shadow: Permission denied

`exit_status` shall convert `self` to a platform-specific exit code,
conveying at least a notion of success or failure.  The return type is
`i32` to match [std::process::exit][] (which probably calls the C
library's `exit` primitive), but (as already documented for
`process::exit`) on "most Unix-like" operating systems, only the low 8
bits of this value are significant.

[std::process::exit]: https://doc.rust-lang.org/std/process/fn.exit.html

## Changes to `main`
[changes-to-main]: #changes-to-main

From the perspective of the code that calls `main`, its signature is
now generic:

``` rust
fn<T: Termination> main() -> T { ... }
```

It is critical that people _writing_ main should not have to treat it
as a generic, though.  Existing code with `fn main() -> ()` should
continue to compile, and code that wants to use the new feature should
be able to write `fn main() -> TermT` for some concrete type `TermT`.

I also don't know whether the code that calls `main` can accept a
generic.  The `lang_start` glue currently receives an unsafe pointer
to `main`, and expects it to have the signature `() -> ()`.  The
abstractly correct new signature is `() -> Termination` but I suspect
that this is currently not possible, pending `impl Trait` return types
or something similar.  `() -> Box<Termination>` probably _is_ possible
but requires a heap allocation, which is undesirable in no-std
contexts.

A solution to both problems is to implement the new semantics entirely
within the compiler.  When it notices that `main` returns anything
other than `()`, it renames the function and injects a shim:

``` rust
fn main() -> () {
    let term = $real_main();
    term.write_diagnostic(std::env::progname(), std::io::LOCAL_STDERR);
    std::process::exit(term.exit_status());
}
```

## `main` in nostd environments
[main-in-nostd-environments]: #main-in-nostd-environments

Some no-std environments do have a notion of processes that run and
then exit, but they may or may not have notions of "exit status" or
"error messages".  In this case, the signature of `main` should be
unchanged, and the shim should simply ignore whichever aspects of
`Termination` don't make sense in context.

There are also environments where returning from `main` constitutes a
_bug_.  If you are implementing an operating system kernel, for
instance, there may be nothing to return to.  Then you want it to be a
compile-time error for `main` to return anything other than `!`.  If
everything is implemented correctly, such environments should be able
to get that effect by omitting all stock impls of `Termination` other
than for `!`.  Perhaps there should also be a compiler hook that
allows such environments to refuse to let you impl Termination
yourself.

## Test functions and doctests
[test-functions-and-doctests]: #test-functions-and-doctests

The harness for `#[test]` functions is very simple; I think it would
be enough to just give `#[test]` functions the same shim that we give
to `main`.  The programmer would be responsible for adjusting their
`#[test]` functions' return types if they want to use `?`, but
existing code would continue to work.

Doctests require a little magic, because you normally don't write the
function head for a doctest yourself, only its body.  This magic
belongs in rustdoc, not in rustc.  When `maketest` sees that it needs
to insert a function head for `main`, it should now write out

``` rust
fn main () -> Result<(), ErrorT> {
   ...
   Ok(())
}
```

for some value of `ErrorT` TBD.  It doesn't need to parse the body of
the test to know whether it should do this; it can just do it
unconditionally.

## Standard impls of Termination
[standard-impls-of-termination]: #standard-impls-of-termination

At least the following implementations of Termination are available in
libstd.  I use the ISO C constants `EXIT_SUCCESS` and `EXIT_FAILURE`
for exposition.  [See below][unix-specific-refinements] for more
discussion of these constants and the `TermStatus` type.

``` rust
impl Termination for ! {
    fn write_diagnostic(&self, progname: &str, stream: &mut Write) -> ()
    { unreachable!(); }
    fn exit_status(&self) -> i32
    { unreachable!(); }
}

impl Termination for () {
    fn write_diagnostic(&self, progname: &str, stream: &mut Write) -> ()
    { }
    fn exit_status(&self) -> i32
    { EXIT_SUCCESS }
}

impl Termination for TermStatus {
    fn write_diagnostic(&self, progname: &str, stream: &mut Write) -> ()
    { }
    fn exit_status(&self) -> i32
    { self.0 }
}

impl<E: Error> Termination for E {
    fn write_diagnostic(&self, progname: &str, stream: &mut Write) -> ()
    {
        // unspecified, but not entirely unlike this:
        if let Some(ref cause) = self.cause() {
            cause.write_diagnostic(progname, stream);
        }
        writeln!(stream, "{}: {}\n", progname, self.description());
    }

    fn exit_status(&self) -> i32
    { EXIT_FAILURE }
}

impl<T: Termination, E: Termination> Termination for Result<T, E> {
    fn write_diagnostic(&self, progname: &str, stream: &mut Write) {
        match *self {
            Ok(ref ok) { ok.write_diagnostic(progname, stream); },
            Err(ref err) { err.write_diagnostic(progname, stream); }
        }
    }
    fn exit_status(&self) -> i32 {
        match *self {
            Ok(ref ok) => ok.exit_status(),
            Err(ref err) => match err.exit_status() {
                0 | EXIT_SUCCESS => EXIT_FAILURE,
                e => e
            }
        }
    }
}
```

The impl for `!` allows programs that intend to run forever to be more
self-documenting: `fn main() -> !` will satisfy the implicit trait
bound on the return type.  It might not be necessary to have the code
in libstd, if the compiler can figure out for itself that `-> !`
satisfies _any_ return type; I have heard that this works in Haskell.
But it's probably good to have it in the reference manual anyway, so
people know they can do that.

The impl for `Result<T,E>` does not allow its `Err` case to produce a
successful exit status.  The technical case for doing this is that it
means you can use `Result<(),()>` to encode success or failure without
printing any messages in either case, and the ergonomics case is that
it's less surprising this way.  I don't _think_ it gets in the way of
anything a realistic program would want to do.

Additional impls of Termination should be added as ergonomics dictate.
However, there probably _shouldn't_ be an impl of Termination for
Option, because there are equally strong arguments for None indicating
success and None indicating failure.  `bool` is similarly ambiguous.
And there probably shouldn't be an impl for `i32` or `u8` either,
because that would permit the programmer to return arbitrary numbers
from `main` without thinking at all about whether they make sense as
exit statuses.

A generic implementation of Termination for anything that is Display
_could_ make sense, but my current opinion is that it is too general
and would make it too easy to get undesired behavior by accident.

## Squelching diagnostics
[squelching-diagnostics]: #squelching-diagnostics

It is fairly common for command-line tools to have a mode, often
triggered by an option `-q` or `--quiet`, which suppresses all output,
*including error messages*.  They still exit unsuccessfully if there
are errors, but they don't print anything at all.  This is for use in
shell control-flow constructs, e.g. `if grep -q blah ...; then ...`
An easy way to facilitate this would be to stabilize a subset of the
`set_panic` feature, say a new function `squelch_errors` or
`silence_stderr` which simply discards all output sent to stderr.

Programs that need to do something more complicated than that are
probably better off printing diagnostics by hand, as is done now.

## Unix-specific refinements
[unix-specific-refinements]: #unix-specific-refinements

The C standard only specifies `0`, `EXIT_SUCCESS` and `EXIT_FAILURE`
as arguments to the [`exit`][exit.3] primitive.  (`EXIT_SUCCESS` is
not guaranteed to have the value 0, but calling `exit(0)` *is*
guaranteed to have the same effect as calling `exit(EXIT_SUCCESS)`;
several versions of the `exit` manpage are incorrect on this point.)
Any other argument has an implementation-defined effect.

Within the Unix ecosystem, `exit` is relied upon to pass values in the
range 0 through 127 (*not* 255) up to the parent process.  There is no
general agreement on the meaning of specific nonzero exit codes, but
there are many contexts that give specific codes a meaning, such as:

* POSIX reserves status 127 for certain internal failures in `system`
  and `posix_spawn` that cannot practically be reported via `errno`.

* [`grep`][grep.1] is specified to exit with status 0 if it found a
   match, 1 if it found no matches, and an unspecified value greater
   than 1 if an error occurred.

* [Automake's support for basic testing][automake-tests] defines status
  77 to mean "test skipped" and status 99 to mean "hard error"
  (I'm not sure precisely what "hard error" is for, but probably
  something like "an error so severe that the entire test run should
  be abandoned").

* The BSDs have, for a long time, provided a header
  [`sysexits.h`][sysexits] that defines a fairly rich set of exit
  codes for general use, but as far as I know these have never seen
  wide adoption.  Of these, `EX_USAGE` (code 64) for command-line
  syntax errors is probably the most useful.

* Rust itself uses status 101 for `panic!`, although arguably it
  should be calling `abort` instead.  (`abort` on Unix systems sends a
  different status to the parent than any value you can pass to `exit`.)

It is unnecessary to put most of this stuff into the Rust stdlib,
especially as some of these conventions contradict each other.
However, the stdlib should not get in the way of a program that
intends to conform to any of the above.  This can be done with the
following refinements:

* The stdlib provides `EXIT_SUCCESS` and `EXIT_FAILURE` constants in
  `std::process` (since `exit` is already there).  These constants do
  _not_ necessarily have the values that the platform's C library
  gives them.  `EXIT_SUCCESS` is always 0.  `EXIT_FAILURE` has the same
  value as the C library gives it, _unless_ the C library gives it the
  value 1, in which case 2 is used instead.

* All of the impls of `Termination` in the stdlib are guaranteed to
  use only `EXIT_SUCCESS` and `EXIT_FAILURE`, with one exception:

* There is a type, provisionally called `TermStatus`, which is a
  newtype over `i32`; on Unix (but not on Windows), creating one from
  a value outside the range 0 ... 255 will panic.  It implements
  `Termination`, passing its value to `exit` and not printing any
  diagnostics.  Using this type, you can generate specific exit codes
  when appropriate, without having to avoid using `?` in `main`.

  (It can't be called `ExitStatus` because that name is already
  taken for the return type of `std::process::Command::status`.)

[exit.3]: http://www.cplusplus.com/reference/cstdlib/exit/
[grep.1]: http://pubs.opengroup.org/onlinepubs/9699919799/utilities/grep.html
[automake-tests]: https://www.gnu.org/software/automake/manual/html_node/Scripts_002dbased-Testsuites.html
[sysexits]: https://www.freebsd.org/cgi/man.cgi?query=sysexits


# How We Teach This
[how-we-teach-this]: #how-we-teach-this

This should be taught alongside the `?` operator and error handling in
general.  The stock `Termination` impls in libstd mean that simple
programs that can fail don't need to do anything special:

``` rust
fn main() -> Result<(), io::Error> {
    let mut stdin = io::stdin();
    let mut raw_stdout = io::stdout();
    let mut stdout = raw_stdout.lock();
    for line in stdin.lock().lines() {
        stdout.write(line?.trim().as_bytes())?;
        stdout.write(b"\n")?;
    }
    stdout.flush()
}
```

Doctest examples can freely use `?` with no extra boilerplate;
`#[test]` examples may need their boilerplate adjusted.

More complex programs, with their own error types, still get
platform-consistent exit status behavior for free, as long as they
implement `Error`.  The Book should describe `Termination` to explain
_how_ `Result`s returned from `main` turn into error messages and exit
statuses, but as a thing that most programs will not need to deal with
directly.

Tutorial examples should probably still begin with `fn main() -> ()`
until the tutorial gets to the point where it starts explaining why
`panic!` and `unwrap` are not for "normal errors".

Discussion of `TermStatus` should be reserved for an advanced-topics
section talking about interoperation with the Unix command-line
ecosystem.

# Drawbacks
[drawbacks]: #drawbacks

Generalizing the return type of `main` complicates libstd and/or the
compiler.  It also adds an additional thing to remember when complete
newbies to the language get to error handling.  On the other hand,
people coming to Rust from other languages may find this _less_
surprising than the status quo.

# Alternatives
[alternatives]: #alternatives

Do nothing; continue to live with the trip hazard, the extra
boilerplate required to comply with platform conventions, and people
using `panic!` to report ordinary errors because it's less hassle.

The [pre-RFC][pre-rfc] included a suggestion to use `catch` instead,
but this still involves extra boilerplate in `main` so I'm not
enthusiastic about it.

# Unresolved Questions
[unresolved]: #unresolved-questions

We need to decide what to call the new trait.  The names proposed in
the pre-RFC thread were `Terminate`, which I like OK but have changed
to `Termination` because return value traits should be nouns, and
`Fallible`, which feels much too general, but could be OK if there
were other uses for it?  Relatedly, it is conceivable that there are
other uses for `Termination` in the existing standard library, but I
can't think of any right now.  (Thread join was mentioned in the
[pre-RFC][pre-rfc], but that can already relay anything that's `Send`,
so I don't see that it adds value there.)

We also need to decide where the trait should live.  One obvious place
is in `std::process`, because that is where `exit(i32)` is; on the
other hand, this is basic enough that it may make sense to put at
least some of it in libcore.

I don't know what impls of `Termination` should be available beyond
the ones listed above, nor do I know what impls should be in libcore.
Most importantly, I do not know whether it is necessary to impl
Termination for `Box<T> where T: Termination`.  It might be that Box's
existing impl of Deref renders this unnecessary.

It may make sense to provide a type alias like

    type MayFail = Result<(), Box<Error>>;

which would mean less boilerplate in the most common case for `main`.
However, this may be too single-purpose for a global-prelude type
alias, and if we ever get return type deduction, that would completely
supersede it, but we'd have burned the name forever.

There is an outstanding proposal to [generalize `?`][try-trait]
(see also RFC issues [#1718][rfc-i1718] and [#1859][rfc-i1859]); I
think it is mostly orthogonal to this proposal, but we should make
sure it doesn't conflict and we should also figure out whether we
would need more impls of `Termination` to make them play well
together.

Most operating systems accept only a scalar exit status, but
[Plan 9][], uniquely (to my knowledge), takes an entire string (see
[`exits(2)`][exits.2]).  Do we care?  If we care, what do we do about
it?

The ergonomics of `?` in general would be improved by autowrapping
fall-off-the-end return values in `Ok` if they're not already
`Result`s, but that's another proposal.

[try-trait]: https://github.com/nikomatsakis/rfcs/blob/try-trait/text/0000-try-trait.md
[rfc-i1718]: https://github.com/rust-lang/rfcs/issues/1718
[rfc-i1859]: https://github.com/rust-lang/rfcs/issues/1859
[Plan 9]: http://www.cs.bell-labs.com/plan9/index.html
[exits.2]: http://plan9.bell-labs.com/magic/man2html/2/exits
