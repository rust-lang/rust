// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Standard library macros
//!
//! This modules contains a set of macros which are exported from the standard
//! library. Each macro is available for use when linking against the standard
//! library.

#[macro_escape];

/// The standard logging macro
///
/// This macro will generically log over a provided level (of type u32) with a
/// format!-based argument list. See documentation in `std::fmt` for details on
/// how to use the syntax, and documentation in `std::logging` for info about
/// logging macros.
///
/// # Example
///
/// ```
/// log!(::std::logging::DEBUG, "this is a debug message");
/// log!(::std::logging::WARN, "this is a warning {}", "message");
/// log!(6, "this is a custom logging level: {level}", level=6);
/// ```
#[macro_export]
macro_rules! log(
    ($lvl:expr, $($arg:tt)+) => ({
        let lvl = $lvl;
        if lvl <= __log_level() {
            format_args!(|args| {
                ::std::logging::log(lvl, args)
            }, $($arg)+)
        }
    })
)

/// A convenience macro for logging at the error log level. See `std::logging`
/// for more information. about logging.
///
/// # Example
///
/// ```
/// # let error = 3;
/// error!("the build has failed with error code: {}", error);
/// ```
#[macro_export]
macro_rules! error(
    ($($arg:tt)*) => (log!(1u32, $($arg)*))
)

/// A convenience macro for logging at the warning log level. See `std::logging`
/// for more information. about logging.
///
/// # Example
///
/// ```
/// # let code = 3;
/// warn!("you may like to know that a process exited with: {}", code);
/// ```
#[macro_export]
macro_rules! warn(
    ($($arg:tt)*) => (log!(2u32, $($arg)*))
)

/// A convenience macro for logging at the info log level. See `std::logging`
/// for more information. about logging.
///
/// # Example
///
/// ```
/// # let ret = 3;
/// info!("this function is about to return: {}", ret);
/// ```
#[macro_export]
macro_rules! info(
    ($($arg:tt)*) => (log!(3u32, $($arg)*))
)

/// A convenience macro for logging at the debug log level. See `std::logging`
/// for more information. about logging.
///
/// # Example
///
/// ```
/// debug!("x = {x}, y = {y}", x=10, y=20);
/// ```
#[macro_export]
macro_rules! debug(
    ($($arg:tt)*) => (if cfg!(not(ndebug)) { log!(4u32, $($arg)*) })
)

/// A macro to test whether a log level is enabled for the current module.
///
/// # Example
///
/// ```
/// # struct Point { x: int, y: int }
/// # fn some_expensive_computation() -> Point { Point { x: 1, y: 2 } }
/// if log_enabled!(std::logging::DEBUG) {
///     let x = some_expensive_computation();
///     debug!("x.x = {}, x.y = {}", x.x, x.y);
/// }
/// ```
#[macro_export]
macro_rules! log_enabled(
    ($lvl:expr) => ({
        let lvl = $lvl;
        lvl <= __log_level() && (lvl != 4 || cfg!(not(ndebug)))
    })
)

/// The entry point for failure of rust tasks.
///
/// This macro is used to inject failure into a rust task, causing the task to
/// unwind and fail entirely. Each task's failure can be reaped as the `~Any`
/// type, and the single-argument form of the `fail!` macro will be the value
/// which is transmitted.
///
/// The multi-argument form of this macro fails with a string and has the
/// `format!` sytnax for building a string.
///
/// # Example
///
/// ```should_fail
/// # #[allow(unreachable_code)];
/// fail!();
/// fail!("this is a terrible mistake!");
/// fail!(4); // fail with the value of 4 to be collected elsewhere
/// fail!("this is a {} {message}", "fancy", message = "message");
/// ```
#[macro_export]
macro_rules! fail(
    () => (
        fail!("explicit failure")
    );
    ($msg:expr) => (
        ::std::rt::begin_unwind($msg, file!(), line!())
    );
    ($fmt:expr, $($arg:tt)*) => ({
        // a closure can't have return type !, so we need a full
        // function to pass to format_args!, *and* we need the
        // file and line numbers right here; so an inner bare fn
        // is our only choice.
        //
        // LLVM doesn't tend to inline this, presumably because begin_unwind_fmt
        // is #[cold] and #[inline(never)] and because this is flagged as cold
        // as returning !. We really do want this to be inlined, however,
        // because it's just a tiny wrapper. Small wins (156K to 149K in size)
        // were seen when forcing this to be inlined, and that number just goes
        // up with the number of calls to fail!()
        #[inline(always)]
        fn run_fmt(fmt: &::std::fmt::Arguments) -> ! {
            ::std::rt::begin_unwind_fmt(fmt, file!(), line!())
        }
        format_args!(run_fmt, $fmt, $($arg)*)
    });
)

/// Ensure that a boolean expression is `true` at runtime.
///
/// This will invoke the `fail!` macro if the provided expression cannot be
/// evaluated to `true` at runtime.
///
/// # Example
///
/// ```
/// // the failure message for these assertions is the stringified value of the
/// // expression given.
/// assert!(true);
/// # fn some_computation() -> bool { true }
/// assert!(some_computation());
///
/// // assert with a custom message
/// # let x = true;
/// assert!(x, "x wasn't true!");
/// # let a = 3; let b = 27;
/// assert!(a + b == 30, "a = {}, b = {}", a, b);
/// ```
#[macro_export]
macro_rules! assert(
    ($cond:expr) => (
        if !$cond {
            fail!("assertion failed: {:s}", stringify!($cond))
        }
    );
    ($cond:expr, $msg:expr) => (
        if !$cond {
            fail!($msg)
        }
    );
    ($cond:expr, $($arg:expr),+) => (
        if !$cond {
            fail!($($arg),+)
        }
    );
)

/// Asserts that two expressions are equal to each other, testing equality in
/// both directions.
///
/// On failure, this macro will print the values of the expressions.
///
/// # Example
///
/// ```
/// let a = 3;
/// let b = 1 + 2;
/// assert_eq!(a, b);
/// ```
#[macro_export]
macro_rules! assert_eq(
    ($given:expr , $expected:expr) => ({
        let given_val = &($given);
        let expected_val = &($expected);
        // check both directions of equality....
        if !((*given_val == *expected_val) &&
             (*expected_val == *given_val)) {
            fail!("assertion failed: `(left == right) && (right == left)` \
                   (left: `{:?}`, right: `{:?}`)", *given_val, *expected_val)
        }
    })
)

/// A utility macro for indicating unreachable code. It will fail if
/// executed. This is occasionally useful to put after loops that never
/// terminate normally, but instead directly return from a function.
///
/// # Example
///
/// ~~~rust
/// struct Item { weight: uint }
///
/// fn choose_weighted_item(v: &[Item]) -> Item {
///     assert!(!v.is_empty());
///     let mut so_far = 0u;
///     for item in v.iter() {
///         so_far += item.weight;
///         if so_far > 100 {
///             return *item;
///         }
///     }
///     // The above loop always returns, so we must hint to the
///     // type checker that it isn't possible to get down here
///     unreachable!();
/// }
/// ~~~
#[macro_export]
macro_rules! unreachable(
    () => (fail!("internal error: entered unreachable code"))
)

/// A standardised placeholder for marking unfinished code. It fails with the
/// message `"not yet implemented"` when executed.
#[macro_export]
macro_rules! unimplemented(
    () => (fail!("not yet implemented"))
)

/// Use the syntax described in `std::fmt` to create a value of type `~str`.
/// See `std::fmt` for more information.
///
/// # Example
///
/// ```
/// format!("test");
/// format!("hello {}", "world!");
/// format!("x = {}, y = {y}", 10, y = 30);
/// ```
#[macro_export]
macro_rules! format(
    ($($arg:tt)*) => (
        format_args!(::std::fmt::format, $($arg)*)
    )
)

/// Use the `format!` syntax to write data into a buffer of type `&mut Writer`.
/// See `std::fmt` for more information.
///
/// # Example
///
/// ```
/// # #[allow(unused_must_use)];
/// use std::io::MemWriter;
///
/// let mut w = MemWriter::new();
/// write!(&mut w, "test");
/// write!(&mut w, "formatted {}", "arguments");
/// ```
#[macro_export]
macro_rules! write(
    ($dst:expr, $($arg:tt)*) => ({
        let dst: &mut ::std::io::Writer = $dst;
        format_args!(|args| { ::std::fmt::write(dst, args) }, $($arg)*)
    })
)

/// Equivalent to the `write!` macro, except that a newline is appended after
/// the message is written.
#[macro_export]
macro_rules! writeln(
    ($dst:expr, $($arg:tt)*) => ({
        let dst: &mut ::std::io::Writer = $dst;
        format_args!(|args| { ::std::fmt::writeln(dst, args) }, $($arg)*)
    })
)

/// Equivalent to the `println!` macro except that a newline is not printed at
/// the end of the message.
#[macro_export]
macro_rules! print(
    ($($arg:tt)*) => (format_args!(::std::io::stdio::print_args, $($arg)*))
)

/// Macro for printing to a task's stdout handle.
///
/// Each task can override its stdout handle via `std::io::stdio::set_stdout`.
/// The syntax of this macro is the same as that used for `format!`. For more
/// information, see `std::fmt` and `std::io::stdio`.
///
/// # Example
///
/// ```
/// println!("hello there!");
/// println!("format {} arguments", "some");
/// ```
#[macro_export]
macro_rules! println(
    ($($arg:tt)*) => (format_args!(::std::io::stdio::println_args, $($arg)*))
)

/// Declare a task-local key with a specific type.
///
/// # Example
///
/// ```
/// use std::local_data;
///
/// local_data_key!(my_integer: int)
///
/// local_data::set(my_integer, 2);
/// local_data::get(my_integer, |val| println!("{}", val.map(|i| *i)));
/// ```
#[macro_export]
macro_rules! local_data_key(
    ($name:ident: $ty:ty) => (
        static $name: ::std::local_data::Key<$ty> = &::std::local_data::Key;
    );
    (pub $name:ident: $ty:ty) => (
        pub static $name: ::std::local_data::Key<$ty> = &::std::local_data::Key;
    );
)

/// Helper macro for unwrapping `Result` values while returning early with an
/// error if the value of the expression is `Err`. For more information, see
/// `std::io`.
#[macro_export]
macro_rules! try(
    ($e:expr) => (match $e { Ok(e) => e, Err(e) => return Err(e) })
)

#[macro_export]
macro_rules! vec(
    ($($e:expr),*) => ({
        let mut temp = ::std::vec_ng::Vec::new();
        $(temp.push($e);)*
        temp
    })
)


/// A macro to select an event from a number of ports.
///
/// This macro is used to wait for the first event to occur on a number of
/// ports. It places no restrictions on the types of ports given to this macro,
/// this can be viewed as a heterogeneous select.
///
/// # Example
///
/// ```
/// let (p1, c1) = Chan::new();
/// let (p2, c2) = Chan::new();
/// # fn long_running_task() {}
/// # fn calculate_the_answer() -> int { 42 }
///
/// spawn(proc() { long_running_task(); c1.send(()) });
/// spawn(proc() { c2.send(calculate_the_answer()) });
///
/// select! (
///     () = p1.recv() => println!("the long running task finished first"),
///     answer = p2.recv() => {
///         println!("the answer was: {}", answer);
///     }
/// )
/// ```
///
/// For more information about select, see the `std::comm::Select` structure.
#[macro_export]
#[experimental]
macro_rules! select {
    (
        $($name:pat = $port:ident.$meth:ident() => $code:expr),+
    ) => ({
        use std::comm::Select;
        let sel = Select::new();
        $( let mut $port = sel.handle(&$port); )+
        unsafe {
            $( $port.add(); )+
        }
        let ret = sel.wait();
        $( if ret == $port.id() { let $name = $port.$meth(); $code } else )+
        { unreachable!() }
    })
}
