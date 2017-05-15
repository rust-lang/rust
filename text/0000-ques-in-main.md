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
`Termination`.  libstd implements this trait for `!`, `()`,
`process::ExitStatus`, `Result<(), E> where E: Error`, and possibly
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
use std::net::{SocketAddrV4, TcpStream, UdpSocket, TcpListener, Ipv4Addr};
let ip = Ipv4Addr::new(127, 0, 0, 1);
let port = 12345;

// The following lines are equivalent modulo possible "localhost" name
// resolution differences
let tcp_s = TcpStream::connect(SocketAddrV4::new(ip, port));
let tcp_s = TcpStream::connect((ip, port));
let tcp_s = TcpStream::connect(("127.0.0.1", port));
let tcp_s = TcpStream::connect(("localhost", port));
let tcp_s = TcpStream::connect("127.0.0.1:12345");
let tcp_s = TcpStream::connect("localhost:12345");

// TcpListener::bind(), UdpSocket::bind() and UdpSocket::send_to()
// behave similarly
let tcp_l = TcpListener::bind("localhost:12345");

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
those environments' conventions.  A typical construction is

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
   due to a system error.
1. We should avoid making life more complicated for people who don't
   care; on the other hand, if the Easiest Thing is also the Right
   Thing according to the platform convention, that is better all
   around.

Goal 1 dictates that the new return type for `main` will be
`Result<T, E>` for some T and E.  To minimize the necessary changes to
existing code that wants to start using `?` in `main`, T should be
allowed to be `()`, but other types in that position may also make
sense.  The appropriate bound for E is unclear; there are plausible
arguments for at least `Error`, `Debug`, and `Display`.  This proposal
starts from the narrowest possibility and provides for only
`Result<() E> where E: Error`.

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

At least the following implementations of Termination are available in
libstd.  I use the ISO C constants `EXIT_SUCCESS` and `EXIT_FAILURE`
for exposition; they are not necesssarily intended to be the exact
values passed to `process::exit`.

``` rust
impl Termination for ! {
    fn report(self) -> i32 { unreachable!(); }
}

impl Termination for () {
    fn report(self) -> i32 { EXIT_SUCCESS }
}

impl Termination for std::process::ExitStatus {
    fn report(self) -> i32 {
        self.code().expect("Cannot use a signal ExitStatus to exit")
    }
}

fn print_diagnostics_for_error<E: Error>(err: &E) {
    // unspecified, but along the lines of:
    if let Some(ref cause) = err.cause() {
        print_diagnostics_for_error(cause);
    }
    eprintln!("{}: {}", get_program_name(), err.description());
}

impl<E: Error> Termination for Result<(), E> {
    fn report(self) -> i32 {
        match self {
            Ok(_) => EXIT_SUCCESS,
            Err(ref err) => {
                print_diagnostics_for_error(err);
                EXIT_FAILURE
            }
        }
    }
}
```

The impl for `!` allows programs that intend to run forever to be more
self-documenting: `fn main() -> !` will satisfy the implicit trait
bound on the return type.  It might not be necessary to have code for
this impl in libstd, since `-> !` satisfies `-> ()`, but it should
appear in the reference manual anyway, so people know they can do
that, and it may also be desirable as a backstop against a `main` that
does somehow return, despite declaring that it doesn't.

The impl for `ExitStatus` allows programs to generate both success and
failure conditions _without_ any errors printed, by returning from
`main`.  This is meant to be used by sophisticated programs that do
all of their own error-message printing themselves.
[See below][exit-status] for more discussion and related changes to
`ExitStatus`.

Additional impls of Termination should be added as ergonomics dictate.
For instance, it may well make sense to provide an impl for
`Result<(), E> where E: Display` or `... where E: Debug`, because
programs may find it convenient to use bare strings as error values,
and application error types are not obliged to be `std::Error`s.
Also, it is unclear to me whether `impl<E: Error> Termination for
Result<(), E>` applies to `Result<(), Box<Error>>`.  If it doesn't, we
will almost certainly need to add `impl<E: Box<Error>> Termination for
Result<(), E>`.  I hope that isn't necessary, because then we might
also need `Rc<Error>` and `Arc<Error>` and `Cow<Error>` and
`RefCell<Error>` and ...

There probably _shouldn't_ be an impl of Termination for Option,
because there are equally strong arguments for None indicating success
and None indicating failure.  `bool` is similarly ambiguous.  And
there probably shouldn't be an impl for `i32` or `u8` either, because
that would permit the programmer to return arbitrary numbers from
`main` without thinking at all about whether they make sense as exit
statuses.

A previous revision of this RFC included an impl for `Result<T, E>
where T: Termination, E: Termination`, but it has been removed from
this version, as some people felt it would allow undesirable behavior.
It can always be added again in the future.

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
    use process::exit;

    sys::init();

    exit(unsafe {
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
        let status = match panic::catch_unwind(main) {
            Ok(term) { term.report() }
            Err(_)   { 101 }
        }
        sys_common::cleanup();
        status
    });
}
```

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

## Test functions and doctests
[test-functions-and-doctests]: #test-functions-and-doctests

The harness for `#[test]` functions will need to be changed, similarly
to how the "start" lang item was changed.  Tests which return
anything whose `report()` method returns a nonzero value should be
considered to have failed.

Doctests require a little magic in rustdoc: when `maketest` sees that
it needs to insert a function head for `main`, it should now write out

``` rust
fn main () -> Result<(), ErrorT> {
   ...
   Ok(())
}
```

for some value of `ErrorT` TBD.  It doesn't need to parse the body of
the test to know whether it should do this; it can just do it
unconditionally.

## New constructors for `ExitStatus`
[exit-status]: #exit-status

As mentioned above, we propose to reuse `process::ExitStatus` as a way
for a program to generate both success and failure conditions
_without_ any errors printed, by returning it from `main`.  To make
this more convenient, we also propose to add the following new
constructors to `ExitStatus`:

``` rust
impl ExitStatus {
    /// Return an ExitStatus value representing success.
    /// (ExitStatus::ok()).success() is guaranteed to be true, and
    /// (ExitStatus::ok()).code() is guaranteed to be Some(0).
    pub fn ok() -> Self;

    /// Return an ExitStatus value representing failure.
    /// (ExitStatus::failure()).success() is guaranteed to be false, and
    /// (ExitStatus::failure()).code() is guaranteed to be Some(n),
    /// for some unspecified n, 1 < n < 64.
    pub fn failure() -> Self;

    /// Return an ExitStatus value representing a specific exit code.
    /// The difference between this method and ExitStatusExt::from_raw
    /// is that this method can only be used to produce ExitStatus
    /// values that will pass unmodified through the operating system
    /// primitive "exit" and "wait" operations on all supported
    /// platforms.  (Conveniently, this is exactly the range of u8.)
    pub fn from_code(code: u8) -> Self;
}
```

The first method's name is `ok` because `success` is already taken.

## Unix-specific refinements
[unix-specific-refinements]: #unix-specific-refinements

The C standard only specifies `0`, `EXIT_SUCCESS` and `EXIT_FAILURE`
as arguments to the [`exit`][exit.3] primitive.  (`EXIT_SUCCESS` is
not guaranteed to have the value 0, but calling `exit(0)` *is*
guaranteed to have the same effect as calling `exit(EXIT_SUCCESS)`;
several versions of the `exit` manpage are incorrect on this point.)
Any other argument has an implementation-defined effect.

Within the Unix ecosystem, `exit` can be relied upon to pass values in
the range 0 through 255 up to the parent process; this is the range
proposed for `ExitStatus::from_code`.  (POSIX says that one should be
able to pass a full C `int` through `exit` as long as the parent uses
`waitid` to retrieve the value, but this is not widely implemented.
Also, values 128 through 255 have historically been avoided because
older implementations were unreliable about zero- rather than
sign-extending in `WEXITSTATUS`.)  There is no general agreement on
the meaning of specific nonzero exit codes, but there are many
contexts that give specific codes a meaning, such as:

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

* All of the implementations of `Termination` in the stdlib, except
  the one for `ExitStatus` itself, are guaranteed to behave as-if they
  use only `ExitStatus::ok()` and `ExitStatus::failure()`.

* The value used by `ExitStatus::failure()` is guaranteed to be
  greater than 1 and less than 64.  This avoids collisions with all of
  the above conventions.  (I recommend we actually use 2, because
  people may think that any larger value has a specific meaning, and
  then waste time trying to find out what it is.)

[exit.3]: http://www.cplusplus.com/reference/cstdlib/exit/
[grep.1]: http://pubs.opengroup.org/onlinepubs/9699919799/utilities/grep.html
[automake-tests]: https://www.gnu.org/software/automake/manual/html_node/Scripts_002dbased-Testsuites.html
[sysexits]: https://www.freebsd.org/cgi/man.cgi?query=sysexits


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
the top of your screen):

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

fn main() -> ExitStatus {
    match parse_args() {
        Err(err) => {
            eprintln!("{}", err);
            ExitStatus::failure()
        },
        Ok(args) => {
            match process(&args.city, &args.data_path) {
                Err(err) => {
                    eprintln!("{}: {}: {}",
                              args.progname, args.data_path, err);
                    ExitStatus::failure()
                },
                Ok(_) => ExitStatus::ok()
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

Tutorial examples should probably still begin with `fn main() -> ()`
until the tutorial gets to the point where it starts explaining why
`panic!` and `unwrap` are not for "normal errors".  The `Termination`
trait should also be explained at that point, to illuminate _how_
`Result`s returned from `main` turn into error messages and exit
statuses, but as a thing that most programs will not need to deal with
directly.

Discussion of `ExitStatus::from_code` should be reserved for an
advanced-topics section talking about interoperation with the Unix
command-line ecosystem.

Doctest examples can freely use `?` with no extra boilerplate;
`#[test]` examples may need their boilerplate adjusted.

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
least some of it in libcore.  Note that the `start` lang item is not
in libcore.

I don't know what impls of `Termination` should be available beyond
the ones listed above, nor do I know what impls should be in libcore.

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
