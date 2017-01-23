- Feature Name: eprintln
- Start Date: 2017-01-23
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

This RFC proposes the addition of two macros to the global prelude,
`eprint!` and `eprintln!`.  These are exactly the same as `print!` and
`println!`, respectively, except that they write to standard error
instead of standard output.

# Motivation
[motivation]: #motivation

This proposal will improve the ergonomics of the Rust language for
development of command-line tools and "back end" / "computational
kernel" programs.  Such programs need to maintain a distinction
between their _primary output_, which will be fed to the next element
in a computational "pipeline", and their _status reports_, which
should go directly to the user.  Conventionally, standard output
should receive the primary output and standard error should receive
status reports.

At present, writing text to standard output is very easy, using the
`print(ln)!` macros, but writing text to standard error is
significantly more work: compare

    println!("out of cheese error: {}", 42);
    writeln!(stderr(), "out of cheese error: {}", 42).unwrap();

The latter may also require the addition of `use std::io::stderr`
and/or `use std::io::Write;` to the top of the file.

Because writing to stderr is more work, and requires introduction of
more concepts, all of the tutorial documentation for the language uses
`println!` for error messages, which teaches bad habits.

# Detailed design
[design]: #detailed-design

Most of the design is already nailed down by the existing `println!`
macro.  It is my intention to clone the existing definition verbatim,
changing only the name and the I/O stream written to.  Anything other
than strict parallelism with `println!` would be surprising and
confusing.

There remain two design decisions to make:

 * Should there be an `eprint!` analogous to `print!`?  (That is, a
   macro that writes formatted text to stderr _without_ appending a
   newline.)  I suspect that it will be rarely used, but I also
   suspect that its absence will be surprising.  Leaving out `eprint!`
   might enable a better choice of name for `eprintln!` (see below).

 * What should the name(s) of the macro(s) be?  There were four
   candidates proposed in [the pre-RFC][pre-rfc]:

   * `eprintln!` and `eprint!` -- easy to type, and two different
     people said they were already using these names in their own
     code; but perhaps too cryptic.

   * `println_err!` and `print_err!` -- less cryptic, but significantly
     more awkward to type; it is the author's personal opinion that
     these names will lead people to continue using `println!` for
     error messages.

   * `errorln!` and possibly also `error!` -- ruled out by the
     [`log` crate][log-crate] already using `error!` for something
     else, IMHO.

   * `errln!` and possibly also `err!` -- `err!` is too likely to be
     taken by a crate already IMHO.  If we could omit the no-newline
     macro, however, I rather like `errln!`.

An [implementation][] (using `eprint(ln)!`) already exists and should
need only trivial revisions after the above decisions are made.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

Since these macros are exactly the same as the existing `print(ln)!`
macros but for writing to stderr, it will not be necessary to teach
how they work.

It will, however, be necessary to add text to the reference manual and
especially to the tutorials explaining the difference between "primary
output" and "status reports", so that programemrs know _when_ to use
them.  All of the existing examples and tutorials should be checked
over for cases where `println!` is being used for a status report, and
all such cases should be changed to use the new macro instead.

# Drawbacks
[drawbacks]: #drawbacks

The usual drawbacks of adding macros to the prelude apply.  In this
case, I think the most significant concern is to choose names that are
unlikely to to conflict with existing library crates.

# Alternatives
[alternatives]: #alternatives

Conceivably, the ergonomics of `writeln!` could be improved to make
this unnecessary.  There are three fundamental problems with that,
though:

1. `writeln!(stderr(), ...)` is always going to be more typing than
   `eprintln!(...)`.  People live with `fprintf(stderr, ...)` in C, so
   perhaps that's not that bad.

1. `writeln!` returns a Result, which must be consumed; this is
   appropriate for the intended core uses of `writeln!`, but means
   tacking `.unwrap()` on the end of every use to print diagnostics
   (if printing diagnostics fails, it is almost always the case that
   there's nothing more sensible to do than crash).

1. `writeln!(stderr(), ...)` is unaffected by `set_panic()` (just as
   `writeln!(stdout(), ...)` is unaffected by `set_print()`).  This is
   arguably a bug.  On the other hand, it is also arguably the Right Thing.

# Unresolved questions
[unresolved]: #unresolved-questions

See discussion above.

[pre-rfc]: https://internals.rust-lang.org/t/extremely-pre-rfc-eprintln/4635/10
[log-crate]: https://crates.io/crates/log
[implementation]: https://github.com/rust-lang/rust/pull/39229/files
