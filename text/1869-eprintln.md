- Feature Name: eprintln
- Start Date: 2017-01-23
- RFC PR: [rust-lang/rfcs#1869](https://github.com/rust-lang/rfcs/pull/1869)
- Rust Issue: [rust-lang/rust#40528](https://github.com/rust-lang/rust/issues/40528)

# Summary
[summary]: #summary

This RFC proposes the addition of two macros to the global prelude,
`eprint!` and `eprintln!`.  These are exactly the same as `print!` and
`println!`, respectively, except that they write to standard error
instead of standard output.

An [implementation][] already exists.

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

Two macros will be added to the global prelude.  `eprint!` is exactly
the same as `print!`, and `eprintln!` is exactly the same as
`println!`, except that both of them write to standard error instead
of standard output.  "Standard error" is defined as "the same place
where `panic!` writes messages."  In particular, using `set_panic` to
change where panic messages go will also affect `eprint!` and
`eprintln!`.

Previous discussion has converged on agreement that both these macros
will be useful, but has not arrived at a consensus about their names.
An executive decision is necessary.  It is the author's opinion that
`eprint!` and `eprintln!` have the strongest case in their favor,
being (a) almost as short as `print!` and `println!`, (b) still
visibly different from them, and (c) the names chosen by several
third-party crate authors who implemented these macros themselves for
internal use.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

We will need to add text to the reference manual, and especially to
the tutorials, explaining the difference between "primary output" and
"status reports", so that programmers know when to use `println!` and
when to use `eprintln!`.  All of the existing examples and tutorials
should be checked over for cases where `println!` is being used for a
status report, and all such cases should be changed to use `eprintln!`
instead; similarly for `print!`.

# Drawbacks
[drawbacks]: #drawbacks

The usual drawbacks of adding macros to the prelude apply.  In this
case, I think the most significant concern is to choose names that are
unlikely to to conflict with existing library crates' _exported_
macros.  (Conversely, _internal_ macros with the same names and
semantics demonstrate that the names chosen are appropriate.)

The names `eprintln!` and `eprint!` are terse, differing only in a
single letter from `println!` and `print!`, and it's not obvious at a
glance what the leading `e` means.  ("This is too cryptic" is the
single most frequently heard complaint from people who don't like
`eprintln!`.)  However, once you do know what it means it is
reasonably memorable, and anyone who is already familiar with stdout
versus stderr is very likely to guess correctly what it means.

There is an increased teaching burden---but that's the wrong way to
look at it.  The Book and the reference manual _should have_ been
teaching the difference between "primary output" and "status reports"
all along.  This is something programmers already need to know in
order to write programs that fit well into the larger ecosystem.  Any
documentation that might be a new programmer's first exposure to the
concept of "standard output" has a duty to explain that there is also
"standard error", and when you should use which.

# Alternatives
[alternatives]: #alternatives

It would be inappropriate to introduce printing-to-stderr macros whose
behavior did not exactly parallel the existing printing-to-stdout
macros; I will not discuss that possibility further.

We could provide only `eprintln!`, omitting the no-newline variant.
Most _error_ messages should be one or more complete lines, so it's
not obvious that we need `eprint!`.  However, standard error is also
the appropriate place to send _progress_ messages, and it is common to
want to print partial lines in progress messages, as this is a natural
way to express "a time-consuming computation is running".
[For example][progress-ex]:

```
Particle        0 of      200: (0.512422, 0.523495, 0.481173)  ( 1184 ms)
Particle        1 of      200: (0.521386, 0.543189, 0.473058)  ( 1202 ms)
Particle        2 of      200: (0.498974, 0.538118, 0.488474)  ( 1146 ms)
Particle        3 of      200: (0.546846, 0.565138, 0.500004)  ( 1171 ms)
Particle        4 of      200: _
```

We could choose different names.  Quite a few other possibilities have
been suggested in the [pre-RFC][] and [RFC][] discussions; they fall
into three broad classes:

 * `error(ln)!` and `err(ln)!` are ruled out as too likely to collide
   with third-party crates.  `error!` in particular is already taken
   by the [`log` crate][log-crate].

 * `println_err!`, `printlnerr!`, `errprintln!`, and several other
   variants on this theme are less terse, but also more typing.  It is
   the author's personal opinion that minimizing additional typing
   here is a Good Thing.  People do live with `fprintf(stderr, ...)`
   in C, but on the other hand there is a lot of sloppy C out there
   that sends its error messages to stdout.  I want to minimize the
   friction in _using_ `eprintln!` once you already know what it means.

   It is also highly desirable to put the distinguishing label at the
   _beginning_ of the macro name, as this makes the difference stand
   out more when skimming code.

 * `aprintln!`, `dprintln!`, `uprintln!`, `println2!`, etc. are not
   less cryptic than `eprintln!`, and the official name of standard
   I/O stream 2 is "standard _error_", even though it's not just for
   errors, so `e` is the best choice.

Finally, we could think of some way to improve the ergonomics of
`writeln!` so that we don't need the new macros at all.  There are
four fundamental problems with that, though:

1. `writeln!(stderr(), ...)` is always going to be more typing than
   `eprintln!(...)`.  (Again, people do live with `fprintf(stderr,
   ...)` in C, but again, minimizing usage friction is highly
   desirable.)

1. On a similar note, use of `writeln!` requires `use std::io::Write`,
   in contrast to C where `#include <stdio.h>` gets you both `printf`
   and `fprintf`.  I am not sure how often this would be the _only_
   use of `writeln!` in complex programs, however.

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

[pre-RFC]: https://internals.rust-lang.org/t/extremely-pre-rfc-eprintln/4635/10
[RFC]: https://github.com/rust-lang/rfcs/pull/1869
[progress-ex]: https://github.com/rust-lang/rfcs/pull/1869#issuecomment-274609380
[log-crate]: https://crates.io/crates/log
[implementation]: https://github.com/rust-lang/rust/pull/39229/files
