- Feature Name: ques_in_main
- Start Date: 2017-02-22
- RFC PR: https://github.com/rust-lang/rfcs/pull/1937
- Rust Issue: https://github.com/rust-lang/rust/issues/43301

# Summary
[summary]: #summary

Allow the `?` operator to be used in `main`, and in `#[test]`
functions and doctests.

To make this possible, the return type of these functions are
generalized from `()` to a new trait, provisionally called
`Termination`.  libstd implements this trait for a set of types
partially TBD (see [list below](#standard-impls-of-termination));
applications can provide impls themselves if they want.

There is no magic added to function signatures in rustc.  If you want
to use `?` in either `main` or a `#[test]` function you have to write
`-> Result<(), ErrorT>` (or whatever) yourself.  Initially, it will
also be necessary to write a hidden function head for any doctest that
wants to use `?`, but eventually (see the
[deployment plan](#deployment-plan) below) the default doctest
template will be adjusted to make this unnecessary most of the time.

[Pre-RFC discussion][pre-rfc].  [Prior RFC issue][old-issue].

[pre-rfc]: https://internals.rust-lang.org/t/rfc-mentoring-opportunity-permit-in-main/4600
[old-issue]: https://github.com/rust-lang/rfcs/issues/1176

# Motivation
[motivation]: #motivation

It is currently not possible to use `?` in `main`, because `main`'s
return type is required to be `()`.  This is a trip hazard for new
users of the language, and complicates "programming in the small".
For example, consider a version of the
[CSV-parsing example from the Rust Book][csv-example]
(I have omitted a chunk of command-line parsing code and the
definition of the Row type, to keep it short):

``` rust
fn main() {
    let argv = env::args();
    let _ = argv.next();
    let data_path = argv.next().unwrap();
    let city = argv.next().unwrap();

    let file = File::open(data_path).unwrap();
    let mut rdr = csv::Reader::from_reader(file);

    for row in rdr.decode::<Row>() {
        let row = row.unwrap();

        if row.city == city {
            println!("{}, {}: {:?}",
                row.city, row.country,
                row.population.expect("population count"));
        }
    }
}
```

The Rust Book uses this as a starting point for a demonstration of how
to do error handing _properly_, i.e. without using `unwrap` and
`expect`.  But suppose this is a program for your own personal use.
You are only writing it in Rust because it needs to crunch an enormous
data file and high-level scripting languages are too slow.  You don't
especially _care_ about proper error handling, you just want something
that works, with minimal programming effort.  You'd like to not have
to remember that this is `main` and you can't use `?`.  You would like
to write instead

``` rust
fn main() -> Result<(), Box<Error>> {
    let argv = env::args();
    let _ = argv.next();
    let data_path = argv.next()?;
    let city = argv.next()?;

    let file = File::open(data_path)?;
    let mut rdr = csv::Reader::from_reader(file);

    for row in rdr.decode::<Row>() {
        let row = row?;

        if row.city == city {
            println!("{}, {}: {:?}",
                row.city, row.country, row.population?);
        }
    }
    Ok(())
}
```

(Just to be completely clear, this is not intended to _reduce_ the
amount of error-handling boilerplate one has to write; only to make it
be the same in `main` as it would be for any other function.)

For the same reason, it is not possible to use `?` in doctests and
`#[test]` functions.  This is only an inconvenience for `#[test]`
functions, same as for `main`, but it's a major problem for doctests,
because doctests are supposed to demonstrate normal usage, as well as
testing functionality.  Taking an
[example from the stdlib][to-socket-addrs]:

``` rust
use std::net::UdpSocket;
let port = 12345;
let mut udp_s = UdpSocket::bind(("127.0.0.1", port)).unwrap(); // XXX
udp_s.send_to(&[7], (ip, 23451)).unwrap(); // XXX
```

The lines marked `XXX` have to use `unwrap`, because a doctest is the
body of a `main` function, but in normal usage, they would be written

``` rust
let mut udp_s = UdpSocket::bind(("127.0.0.1", port))?;
udp_s.send_to(&[7], (ip, 23451))?;
```

and that's what the documentation _ought_ to say.  Documentation
writers can work around this by including their own `main` as
hidden code, but they shouldn't have to.

On a related note, `main` returning `()` means that short-lived
programs, designed to be invoked from the Unix shell or a similar
environment, have to contain extra boilerplate in order to comply with
those environments' conventions, and must ignore the dire warnings
about destructors not getting run in the documentation for
[`process::exit`][process-exit].  (In particular, one might be
concerned that the program below will not properly flush and close
`io::stdout`, and/or will fail to detect delayed write failures on
`io::stdout`.)  A typical construction is

``` rust
fn inner_main() -> Result<(), ErrorT> {
    // ... stuff which may fail ...
    Ok(())
}

fn main() -> () {
    use std::process::exit;
    use libc::{EXIT_SUCCESS, EXIT_FAILURE};

    exit(match inner_main() {
        Ok(_) => EXIT_SUCCESS,

        Err(ref err) => {
            let progname = get_program_name();
            eprintln!("{}: {}\n", progname, err);

            EXIT_FAILURE
        }
    })
}
```

These problems can be solved by generalizing the return type of `main`
and test functions.

[csv-example]: https://doc.rust-lang.org/book/error-handling.html#case-study-a-program-to-read-population-data
[to-socket-addrs]: https://doc.rust-lang.org/std/net/trait.ToSocketAddrs.html
[process-exit]: https://doc.rust-lang.org/std/process/fn.exit.html

# Detailed design
[design]: #detailed-design

The design goals for this new feature are, in decreasing order of
importance:

1. The `?` operator should be usable in `main`, `#[test]` functions,
   and doctests.  This entails these functions now returning a richer
   value than `()`.
1. Existing code with `fn main() -> ()` should not break.
1. Errors returned from `main` in a hosted environment should
   *not* trigger a panic, consistent with the general language
   principle that panics are only for bugs.
1. We should take this opportunity to increase consistency with
   platform conventions for process termination. These often include
   the ability to pass an "exit status" up to some outer environment,
   conventions for what that status means, and an expectation that a
   diagnostic message will be generated when a program fails
   due to a system error.  However, we should not make things more
   complicated for people who don't care.

Goal 1 dictates that the new return type for `main` will be
`Result<T, E>` for some T and E.  To minimize the necessary changes to
existing code that wants to start using `?` in `main`, T should be
allowed to be `()`, but other types in that position may also make
sense.  The appropriate bound for E is unclear; there are plausible
arguments for at least Error, Debug, and Display.  This proposal
selects Display, largely because application error types are not
obliged to implement Error.

To achieve goal 2 at the same time as goal 1, `main`'s return type
must be allowed to vary from program to program.  This can be dealt
with by making the `start` lang item polymorphic (as
[described below](#changes-to-lang-start)) over a
trait which both `()` and `Result<(), E>` implement, and similarly for
[doctests](#changes-to-doctests) and
[`#[test]` functions](#changes-to-the-test-harness).

Goals 3 and 4 are largely a matter of quality of implementation; at
the level of programmer-visible interfaces, people who don't care are
well-served by not breaking existing code (which is goal 2) and by
removing a way in which `main` is not like other functions (goal 1).

## The `Termination` trait
[the-termination-trait]: #the-termination-trait

When `main` returns a nontrivial value, the runtime needs to know two
things about it: what error message, if any, to print, and what value
to pass to `std::process::exit`.  These are naturally encapsulated in
a trait, which we are tentatively calling `Termination`, with this
signature:

``` rust
trait Termination {
    fn report(self) -> i32;
}
```

`report` is a call-once function; it consumes self.  The runtime
guarantees to call this function after `main` returns, but at a point
where it is still safe to use `eprintln!` or `io::stderr()` to print
error messages.  `report` is not _required_ to print error messages,
and if it doesn't, nothing will be printed.  The value it returns will
be passed to `std::process::exit`, and shall convey at least a notion
of success or failure.  The return type is `i32` to match
[std::process::exit][] (which probably calls the C library's `exit`
primitive), but (as already documented for `process::exit`) on "most
Unix-like" operating systems, only the low 8 bits of this value are
significant.

[std::process::exit]: https://doc.rust-lang.org/std/process/fn.exit.html

## Standard impls of Termination
[standard-impls-of-termination]: #standard-impls-of-termination

At least the following implementations of Termination will be added to
libstd.  (Code samples below use the constants `EXIT_SUCCESS` and
`EXIT_FAILURE` for exposition;
[see below](#values-used-for-success-and-failure) for discussion of
what the actual numeric values should be.)  The first two are
essential to the proposal:

``` rust
impl Termination for () {
    fn report(self) -> i32 { EXIT_SUCCESS }
}
```

This preserves backward compatibility: all existing programs, with
`fn main() -> ()`, will still satisfy the new requirement (which is
effectively `fn main() -> impl Termination`, although the proposal
does not actually depend on impl-trait return types).

``` rust
impl<T: Termination, E: Display> Termination for Result<T, E> {
    fn report(self) -> i32 {
        match self {
            Ok(val) => val.report(),
            Err(ref err) => {
                print_diagnostics_for_error(err);
                EXIT_FAILURE
            }
        }
    }
}
```

This enables the use of `?` in `main`.  The type bound is somewhat
more general than the minimum: we accept any type that satisfies
Termination in the Ok position, not just `()`.  This is because, in
the presence of application impls of Termination, it would be
surprising if `fn main() -> FooT` was acceptable but `fn main()
-> Result<FooT, ErrT>` wasn't, or vice versa.  On the Err side, any
displayable type is acceptable, but its value does not affect the exit
status; this is because it would be surprising if an apparent error
return could produce a successful exit status.  (This restriction can
always be relaxed later.)

Note that `Box<T>` is Display if T is Display, so special treatment of
`Box<Error>` is not necessary.

Two additional impls are not strictly necessary, but are valuable for
concrete known usage scenarios:

``` rust
impl Termination for ! {
    fn report(self) -> i32 { unreachable!(); }
}
```

This allows programs that intend to run forever to be more
self-documenting: `fn main() -> !` will satisfy the bound on main's
return type.  It might not be necessary to have code for this impl in
libstd, since `-> !` satisfies `-> ()`, but it should appear in the
reference manual anyway, so people know they can do that, and it may
be desirable to include the code as a backstop against a `main` that
does somehow return, despite declaring that it doesn't.

``` rust
impl Termination for bool {
    fn report(self) -> i32 {
        if (self) { EXIT_SUCCESS } else { EXIT_FAILURE }
    }
}
```

This impl allows programs to generate both success and failure
conditions for their outer environment _without_ printing any
diagnostics, by returning the appropriate values from `main`, possibly
while also using `?` to report error conditions where diagnostics
_should_ be printed.  It is meant to be used by sophisticated programs
that do all, or nearly all, of their own error-message printing
themselves, instead of calling `process::exit` themselves.

The detailed behavior of `print_diagnostics_for_error` is left
unspecified, but it is guaranteed to write diagnostics to `io::stderr`
that include the `Display` text for the object it is passed, and
without unconditionally calling `panic!`.  When the object it is
passed implements `Error` as well as `Display`, it should follow the
`cause` chain if there is one (this may necessitate a separate
Termination impl for `Result<_, Error>`, but that's an implementation
detail).

## Changes to `lang_start`
[changes-to-lang-start]: #changes-to-lang-start

The `start` "lang item", the function that calls `main`, takes the
address of `main` as an argument.  Its signature is currently

``` rust
#[lang = "start"]
fn lang_start(main: *const u8, argc: isize, argv: *const *const u8) -> isize
```

It will need to become generic, something like

``` rust
#[lang = "start"]
fn lang_start<T: Termination>
    (main: fn() -> T, argc: isize, argv: *const *const u8) -> !
```

(Note: the current `isize` return type is incorrect.  As is, the
correct return type is `libc::c_int`.  We can avoid the entire issue by
requiring `lang_start` to call `process::exit` or equivalent itself;
this also moves one step toward not depending on the C runtime.)

The implementation for typical "hosted" environments will be something
like

``` rust
#[lang = "start"]
fn lang_start<T: Termination>
    (main: fn() -> T, argc: isize, argv: *const *const u8) -> !
{
    use panic;
    use sys;
    use sys_common;
    use sys_common::thread_info;
    use thread::Thread;

    sys::init();

    sys::process::exit(unsafe {
        let main_guard = sys::thread::guard::init();
        sys::stack_overflow::init();

        // Next, set up the current Thread with the guard information we just
        // created. Note that this isn't necessary in general for new threads,
        // but we just do this to name the main thread and to give it correct
        // info about the stack bounds.
        let thread = Thread::new(Some("main".to_owned()));
        thread_info::set(main_guard, thread);

        // Store our args if necessary in a squirreled away location
        sys::args::init(argc, argv);

        // Let's run some code!
        let exitcode = panic::catch_unwind(|| main().report())
            .unwrap_or(101);

        sys_common::cleanup();
        exitcode
    });
}
```

## Changes to doctests
[changes-to-doctests]: #changes-to-doctests

Simple doctests form the body of a `main` function, so they require
only a small modification to rustdoc: when `maketest` sees that it
needs to insert a function head for `main`, it will now write out

``` rust
fn main () -> Result<(), ErrorT> {
   ...
   Ok(())
}
```

for some value of `ErrorT` to be worked out
[during deployment](#deployment-plan).  This head will work correctly
for function bodies without any uses of `?`, so rustdoc does not need
to parse each function body; it can use this head unconditionally.

If the doctest specifies its own function head for `main` (visibly or
invisibly), then it is the programmer's responsibility to give it an
appropriate type signature, as for regular `main`.

## Changes to the `#[test]` harness
[changes-to-the-test-harness]: #changes-to-the-test-harness

The appropriate semantics for test functions with rich return values
are straightforward: Call the `report` method on the value returned.
If `report` returns a nonzero value, the test has failed.
(Optionally, honor the Automake convention that exit code 77 means
"this test cannot meaningfully be run in this context.")

The required changes to the test harness are more complicated, because
it supports six different type signatures for test functions:

``` rust
pub enum TestFn {
    StaticTestFn(fn()),
    StaticBenchFn(fn(&mut Bencher)),
    StaticMetricFn(fn(&mut MetricMap)),
    DynTestFn(Box<FnBox<()>>),
    DynMetricFn(Box<for<'a> FnBox<&'a mut MetricMap>>),
    DynBenchFn(Box<TDynBenchFn + 'static>),
}
```

All of these need to be generalized to allow any return type that
implements Termination.  At the same time, it still needs to be
possible to put TestFn instances into a static array.

For the static cases, we can avoid changing the test harness at all
with a built-in macro that generates wrapper functions: for example,
given

``` rust
#[test]
fn test_the_thing() -> Result<(), io::Error> {
    let state = setup_the_thing()?; // expected to succeed
    do_the_thing(&state)?;          // expected to succeed
}

#[bench]
fn bench_the_thing(b: &mut Bencher) -> Result<(), io::Error> {
    let state = setup_the_thing()?;
    b.iter(|| {
        let rv = do_the_thing(&state);
        assert!(rv.is_ok(), "do_the_thing returned {:?}", rv);
    });
}
```

after macro expansion we would have

``` rust
fn test_the_thing_inner() -> Result<(), io::Error> {
    let state = setup_the_thing()?; // expected to succeed
    do_the_thing(&state)?;          // expected to succeed
}

#[test]
fn test_the_thing() -> () {
    let rv = test_the_thing_inner();
    assert_eq!(rv.report(), 0);
}

fn bench_the_thing_inner(b: &mut Bencher) -> Result<(), io::Error> {
    let state = setup_the_thing()?;
    b.iter(|| {
        let rv = do_the_thing(&state);
        assert!(rv.is_ok(), "do_the_thing returned {:?}", rv);
    });
}

#[bench]
fn bench_the_thing(b: &mut Bencher) -> () {
    let rv = bench_the_thing_inner();
    assert_eq!(rv.report(), 0);
}
```

and similarly for StaticMetricFn (no example shown because I cannot
find any actual _uses_ of MetricMap anywhere in the stdlib, so I
don't know what a use looks like).

We cannot synthesize wrapper functions like this for dynamic tests.
We could use trait objects to allow the harness to call
`Termination::report` anyway: for example, assuming that
`runtest::run` returns a Termination type, we would have something
like

``` rust
pub fn make_test_closure(config: &Config, testpaths: &TestPaths)
        -> test::TestFn {
    let config = config.clone();
    let testpaths = testpaths.clone();
    test::DynTestFn(Box::new(move |()| -> Box<Termination> {
       Box::new(runtest::run(config, &testpaths))
    }))
}
```

But this is not that much of an improvement on just checking the
result inside the closure:

``` rust
pub fn make_test_closure(config: &Config, testpaths: &TestPaths)
        -> test::TestFn {
    let config = config.clone();
    let testpaths = testpaths.clone();
    test::DynTestFn(Box::new(move |()| {
       let rv = runtest::run(config, &testpaths);
       assert_eq(rv.report(), 0);
    }))
}
```

Considering also that dynamic tests are not documented and rarely used
(the only cases I can find in the stdlib are as an adapter mechanism
within libtest itself, and the compiletest harness) I think it makes
most sense to not support rich return values from dynamic tests for now.

## `main` in nostd environments
[main-in-nostd-environments]: #main-in-nostd-environments

Some no-std environments do have a notion of processes that run and
then exit, but do not have a notion of "exit status".  In this case,
`process::exit` probably already ignores its argument, so `main` and
the `start` lang item do not need to change.  Similarly, in an
environment where there is no such thing as an "error message",
`io::stderr()` probably already points to the bit bucket, so `report`
functions can go ahead and use `eprintln!` anyway.

There are also environments where
[returning from `main` constitutes a _bug_.][divergent-main] If you
are implementing an operating system kernel, for instance, there may
be nothing to return to.  Then you want it to be a compile-time error
for `main` to return anything other than `!`.  If everything is
implemented correctly, such environments should be able to get that
effect by omitting all stock impls of `Termination` other than for
`!`.  Perhaps there should also be a compiler hook that allows such
environments to refuse to let you impl Termination yourself.

[divergent-main]: https://internals.rust-lang.org/t/allowing-for-main-to-be-divergent-in-embedded-environments/4717

## The values used for `EXIT_SUCCESS` and `EXIT_FAILURE` by standard impls of Termination
[values-used-for-success-and-failure]: #values-used-for-success-and-failure

The C standard only specifies `0`, `EXIT_SUCCESS` and `EXIT_FAILURE`
as arguments to the [`exit`][exit.3] primitive.  It does not require
`EXIT_SUCCESS` to be zero, but it does require `exit(0)` to have the
same *effect* as `exit(EXIT_SUCCESS)`.  POSIX does require
`EXIT_SUCCESS` to be zero, and the only historical C implementation I
am aware of where `EXIT_SUCCESS` was _not_ zero was for VAX/VMS, which
is probably not a relevant portability target for Rust.
`EXIT_FAILURE` is only required (implicitly in C, explicitly in POSIX)
to be nonzero.  It is _usually_ 1; I have not done a thorough survey
to find out if it is ever anything else.

Within both the Unix and Windows ecosystems, there are several
different semi-conflicting conventions that assign meanings to
specific nonzero exit codes.  It might make sense to include some
support for these conventions in the stdlib (e.g. with a module that
provides the same constants as [`sysexits.h`][sysexits]), but that is
beyond the scope of this RFC.  What _is_ important, in the context of
this RFC, is for the standard impls of Termination to not get in the
way of any program that wants to use one of those conventions.
Therefore I am proposing that all the standard impls' `report`
functions should use 0 for success and 2 for failure.  (It is
important not to use 1, even though `EXIT_FAILURE` is usually 1,
because some existing programs (notably [`grep`][grep.1]) give 1 a
specific meaning; as far as I know, 2 has no specific meaning
anywhere.)

[exit.3]: http://www.cplusplus.com/reference/cstdlib/exit/
[sysexits]: https://www.freebsd.org/cgi/man.cgi?query=sysexits
[grep.1]: http://pubs.opengroup.org/onlinepubs/9699919799/utilities/grep.html

# Deployment Plan
[deployment-plan]: #deployment-plan

This is a complicated feature; it needs two mostly-orthogonal feature
gates and a multi-phase deployment sequence.

The first feature gate is `#![feature(rich_main_return)]`, which must
be enabled to write a main function, test function, or doctest that
returns something other than `()`.  This is not a normal unstable-feature
annotation; it has more in common with a lint check and may need to be
implemented as such.  It will probably be possible to stabilize this
feature quickly—one or two releases after it is initially implemented.

The second feature gate is `#![feature(termination_trait)]`, which
must be enabled to make *explicit* use of the Termination trait,
either by writing new impls of it, or by calling `report` directly.
However, it is *not* necessary to enable this feature gate to merely
return rich values from main, test functions, etc (because in that
case the call to `report` is in stdlib code).  I *think* this is the
semantic of an ordinary unstable-feature annotation on Termination,
with appropriate use-this annotations within the stdlib.  This feature
should not be stabilized for at least a full release after the
stabilization of the `rich_main_return` feature, because it has more
complicated backward compatibility implications, and because it's not
going to be used very often so it will take longer to gain experience
with it.

In addition to these feature gates, rustdoc will initially not change
its template for `main`.  In order to use `?` in a doctest, at first
it will be necessary for the doctest to specify its own function head.
For instance, the `ToSocketAddrs` example from the
["motivation" section](#motivation) will initially need to be written

``` rust
/// # #![feature(rich_main_return)]
/// # fn main() -> Result<(), io::Error> {
/// use std::net::UdpSocket;
/// let port = 12345;
/// let mut udp_s = UdpSocket::bind(("127.0.0.1", port))?;
/// udp_s.send_to(&[7], (ip, 23451))?;
/// # Ok(())
/// # }
```

After enough doctests have been updated, we can survey them to learn
what the most appropriate default function signature for doctest
main is, and only then should rustdoc's template be changed.
(Ideally, this would happen at the same time that `rich_main_return`
is stabilized, but it might need to wait longer, depending on how
enthusiastic people are about changing their doctests.)

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

This should be taught alongside the `?` operator and error handling in
general.  The stock `Termination` impls in libstd mean that simple
programs that can fail don't need to do anything special.

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

Programs that care about the exact structure of their error messages
will still need to use `main` primarily for error reporting.
Returning to the [CSV-parsing example][csv-example], a "professional"
version of the program might look something like this (assume all of
the boilerplate involved in the definition of `AppError` is just off
the top of your screen; also assume that `impl Termination for bool`
is available):

``` rust
struct Args {
    progname: String,
    data_path: PathBuf,
    city: String
}

fn parse_args() -> Result<Args, AppError> {
    let argv = env::args_os();
    let progname = argv.next().into_string()?;
    let data_path = PathBuf::from(argv.next());
    let city = argv.next().into_string()?;
    if let Some(_) = argv.next() {
        return Err(UsageError("too many arguments"));
    }
    Ok(Args { progname, data_path, city })
}

fn process(city: &String, data_path: &Path) -> Result<Args, AppError> {
    let file = File::open(args.data_path)?;
    let mut rdr = csv::Reader::from_reader(file);

    for row in rdr.decode::<Row>() {
        let row = row?;

        if row.city == city {
            println!("{}, {}: {:?}",
                row.city, row.country, row.population?);
        }
    }
}

fn main() -> bool {
    match parse_args() {
        Err(err) => {
            eprintln!("{}", err);
            false
        },
        Ok(args) => {
            match process(&args.city, &args.data_path) {
                Err(err) => {
                    eprintln!("{}: {}: {}",
                              args.progname, args.data_path, err);
                    false
                },
                Ok(_) => true
            }
        }
    }
}
```

and a detailed error-handling tutorial could build that up from the
quick-and-dirty version.  Notice that this is not using `?` in main,
but it _is_ using the generalized `main` return value.  The
`catch`-block feature (part of [RFC #243][rfc243] along with `?`;
[issue #39849][issue39849]) may well enable shortening this `main`
and/or putting `parse_args` and `process` back inline.

Tutorial examples should still begin with `fn main() -> ()` until the
tutorial gets to the point where it starts explaining why `panic!` and
`unwrap` are not for "normal errors".  The `Termination` trait should
also be explained at that point, to illuminate _how_ `Result`s
returned from `main` turn into error messages and exit statuses, but
as a thing that most programs will not need to deal with directly.

Once the doctest default template is changed, doctest examples can
freely use `?` with no extra boilerplate, but `#[test]` examples
involving `?` will need their boilerplate adjusted.

[rfc243]: https://github.com/rust-lang/rfcs/blob/master/text/0243-trait-based-exception-handling.md
[issue39849]: https://github.com/rust-lang/rust/issues/39849

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
"Template projects" (e.g. [quickstart][]) mean that one need not write
out all the boilerplate by hand, but it's still there.

[quickstart]: https://github.com/rusttemplates/quickstart

# Unresolved Questions
[unresolved]: #unresolved-questions

We need to decide what to call the new trait.  The names proposed in
the pre-RFC thread were `Terminate`, which I like OK but have changed
to `Termination` because value traits should be nouns, and `Fallible`,
which feels much too general, but could be OK if there were other uses
for it?  Relatedly, it is conceivable that there are other uses for
`Termination` in the existing standard library, but I can't think of
any right now.  (Thread join was mentioned in the [pre-RFC][pre-rfc],
but that can already relay anything that's `Send`, so I don't see that
it adds value there.)

We may discover during the deployment process that we want more impls
for Termination.  The question of what type rustdoc should use for
its default `main` template is explicitly deferred till during
deployment.

Some of the components of this proposal may belong in libcore, but
note that the `start` lang item is not in libcore.  It should not be a
problem to move pieces from libstd to libcore later.

It would be nice if we could figure out a way to enable use of `?` in
_dynamic_ test-harness tests, but I do not think this is an urgent problem.

All of the code samples in this RFC need to be reviewed for
correctness and proper use of idiom.

# Related Proposals
[related-proposals]: #related-proposals

This proposal formerly included changes to `process::ExitStatus`
intended to make it usable as a `main` return type.  That has now been
spun off as its own [pre-RFC][exit-status-pre], so that we can take our
time to work through the portability issues involved with going beyond
C's simple success/failure dichotomy without holding up this project.

There is an outstanding proposal to [generalize `?`][try-trait]
(see also RFC issues [#1718][rfc-i1718] and [#1859][rfc-i1859]); I
think it is mostly orthogonal to this proposal, but we should make
sure it doesn't conflict and we should also figure out whether we
would need more impls of `Termination` to make them play well
together.

There is also an outstanding proposal to improve the ergonomics of
`?`-using functions by
[autowrapping fall-off-the-end return values in `Ok`][autowrap-return];
it would play well with this proposal, but is not necessary nor does
it conflict.

[exit-status-pre]: https://internals.rust-lang.org/t/mini-pre-rfc-redesigning-process-exitstatus/5426
[try-trait]: https://github.com/nikomatsakis/rfcs/blob/try-trait/text/0000-try-trait.md
[rfc-i1718]: https://github.com/rust-lang/rfcs/issues/1718
[rfc-i1859]: https://github.com/rust-lang/rfcs/issues/1859
[autowrap-return]: https://internals.rust-lang.org/t/pre-rfc-throwing-functions/5419
