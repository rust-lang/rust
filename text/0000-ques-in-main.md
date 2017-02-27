- Feature Name: ques_in_main
- Start Date: 2017-02-22
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Allow the `?` operator to be used in `main` and in `#[test]` functions

To make this possible, the return type of these functions are
generalized from `()` to a new trait, provisionally called
`Termination`.  libstd implements this trait for `!`, `()`, `bool`,
`Error`, `Result<T, E> where T: Termination, E: Termination`,
and possibly other types TBD.  Applications can provide impls
themselves if they want.

There is no magic added to function signatures.  If you want to use
`?` in either `main` or a `#[test]` you have to write `-> Result<(),
ErrorT>` (or whatever) yourself.  (TBD: It may make sense to provide a
type alias like

    type MayFail = Result<(), Box<Error>>;

which would mean less boilerplate in the most common case for `main`.)

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

```rust
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
   functions).  This entails these functions now returning a richer
   value than `()`.
1. Existing code with `fn main() -> ()` should not break.
1. Errors returned from `main` in a hosted environment should
   **not** trigger a panic, consistent with the general language
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
`Result<T,E>` for some T and E.

To minimize the necessary changes to existing code that wants to start
using `?` in `main`, T should be _allowed_ to be `()`, but other types
in that position may also make sense.  For instance, many
implementations of the [`grep`][grep] utility exit with status 0 when
matches are found, 1 when no matches are found, and 2 when I/O errors
occurred; it would be natural to express that using `Result<bool,
io::Error>`.

The boilerplate shown [above][motivation] will work for any E that is
`Display`, but a tighter bound (e.g. `Error` or `Box<Error>`) might
make implementation more convenient and/or facilitate better error
messages.

[grep]: http://man7.org/linux/man-pages/man1/grep.1.html

## The `Termination` trait
[the-termination-trait]: #the-termination-trait

When `main` returns a nontrivial value, the runtime needs to know two
things about it: what error message, if any, to print, and what value
to pass to the platform's equivalent of [`exit(3)`][exit].  These are
naturally encapsulated in a trait, which we are tentatively calling
`Termination`, with this signature:

```rust
trait Termination {
    fn write_diagnostic(&self, progname: &str, stream: &mut Write) -> ();
    fn exit_status(&self) -> i32;
}
```

The canonical home of this trait is `std::process`.

`write_diagnostic` shall write a complete diagnostic message for
`self` to its `stream` argument.  If it produces no output, there will
be no output.  `stream` will normally be the same stream that `panic`
messages from the main thread would go to, which is normally "standard
error".

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

(It would probably be a good idea to reexport `libc::EXIT_SUCCESS` and
`libc::EXIT_FAILURE` from `std::process`.)

Embedded operating systems that have no notion of process exit status
shall ignore the argument to `process::exit`, but shall still call
`write_diagnostic`.  Embedded operating systems where returning from
`main` constitutes a _bug_ shall not provide `Termination` at all;
when targeting such systems, the compiler shall insist that the
signature of `main` be `() -> !` instead of `() -> ()`, and `?` shall
continue to be unusable in `main`.

[std::process::exit]: https://doc.rust-lang.org/std/process/fn.exit.html

## Standard impls of Termination
[standard-impls-of-termination]: #standard-impls-of-termination

At least the following implementations of Termination are available in libstd:

```rust
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

impl Termination for bool {
    fn write_diagnostic(&self, progname: &str, stream: &mut Write) -> ()
    { }
    fn exit_status(&self) -> i32
    { if *self { EXIT_SUCCESS } else { EXIT_FAILURE } }

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
            Ok(ref ok) { ok.write_diagnostic(progname, stream); }
            Err(ref err) { err.write_diagnostic(progname, stream); }
        }
    }
    fn exit_status(&self) -> i32 {
        match *self {
            Ok(ref ok) { ok.exit_status() }
            Err(ref err) { err.exit_status() }
        }
    }
}
```

If the platform permits, the actual exit status used for `false` will
be different from the exit status used for an `Error`.  For instance,
on Unix-like platforms `false` is mapped to status 1 and `Error`s are
mapped to status 2.  (This is the convention used by many
implementations of `grep`, allowing it to distinguish "no error but no
matches found" from "an I/O error occurred".)

(If we wanted to gild the lily, we could map `io::Error`s to specific
codes from [`sysexits.h`][sysexits], but I don't know that we should
bother, given how few other programs do.)

Additional impls of Termination should be added as ergonomics dictate.
However, there probably _shouldn't_ be an impl of Termination for
Option, because there are equally strong arguments for None indicating
success and None indicating failure.  And there probably shouldn't be
an impl for i32 or i8 either, because that would permit the programmer
to return arbitrary numbers from `main` without thinking at all about
whether they make sense as exit statuses.

[exit]: https://linux.die.net/man/3/exit
[sysexits]: http://www.unix.com/man-pages.php?os=freebsd&section=3&query=sysexits

## Implementation issues
[implementation-issues]: #implementation-issues

The tricky part of implementation is impedance matching between `main`
and `lang_start`.  `lang_start` currently receives an unsafe pointer
to `main` and expects it to have the signature `() -> ()`.  The
abstractly correct new signature is `() -> &Termination` but I suspect
that this is currently not possible, pending `impl Trait` return types
or something similar.

Failing that, we _could_ implement the new semantics entirely within
the compiler.  When it noticed that `main` returns anything other than
`()`, it would rename the function `inner_main` and inject a shim:

    fn main() -> () {
        let term = inner_main();
        term.write_diagnostic(std::env::progname(), std::io::stderr());
        std::process::exit(term.exit_status());
    }

Similarly for `#[test]` functions.  Everything else would then be
straightforward additions to libstd and/or libcore.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

This should be taught alongside the `?` operator and error handling in
general.  The stock `Termination` impls in libstd mean that simple
programs that can fail don't need to do anything special:

```rust
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

More complex programs, with their own error types, still get
platform-consistent exit status behavior for free, as long as they
implement `Error`.  The Book should describe `Termination` to explain
_how_ `Result`s returned from `main` turn into error messages and exit
statuses, but as a thing that most programs will not need to deal with
directly.

Tutorial examples should probably still begin with `fn main() -> ()`
until the tutorial gets to the point where it starts explaining why
`panic!` and `unwrap` are not for "normal errors".

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
enthusiastic about it.  Also, `catch` doesn't seem to be happening
anytime soon.

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

I don't know what impls of `Termination` should be available beyond
the ones listed above, nor do I know what impls should be in libcore.
Most importantly, I do not know whether it is necessary to impl
Termination for `Box<T> where T: Termination`.  It might be that Box's
existing impl of Deref renders this unnecessary.

The `MayFail` type alias seems helpful, but also maybe a little too
single-purpose.  And if we ever get return type deduction, that would
completely supersede it, but we'd have burned the name forever.

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
