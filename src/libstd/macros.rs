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

#![macro_escape]

/// The entry point for failure of rust tasks.
///
/// This macro is used to inject failure into a rust task, causing the task to
/// unwind and fail entirely. Each task's failure can be reaped as the
/// `Box<Any>` type, and the single-argument form of the `fail!` macro will be
/// the value which is transmitted.
///
/// The multi-argument form of this macro fails with a string and has the
/// `format!` syntax for building a string.
///
/// # Example
///
/// ```should_fail
/// # #![allow(unreachable_code)]
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
        match (&($given), &($expected)) {
            (given_val, expected_val) => {
                // check both directions of equality....
                if !((*given_val == *expected_val) &&
                     (*expected_val == *given_val)) {
                    fail!("assertion failed: `(left == right) && (right == left)` \
                           (left: `{}`, right: `{}`)", *given_val, *expected_val)
                }
            }
        }
    })
)

/// Ensure that a boolean expression is `true` at runtime.
///
/// This will invoke the `fail!` macro if the provided expression cannot be
/// evaluated to `true` at runtime.
///
/// Unlike `assert!`, `debug_assert!` statements can be disabled by passing
/// `--cfg ndebug` to the compiler. This makes `debug_assert!` useful for
/// checks that are too expensive to be present in a release build but may be
/// helpful during development.
///
/// # Example
///
/// ```
/// // the failure message for these assertions is the stringified value of the
/// // expression given.
/// debug_assert!(true);
/// # fn some_expensive_computation() -> bool { true }
/// debug_assert!(some_expensive_computation());
///
/// // assert with a custom message
/// # let x = true;
/// debug_assert!(x, "x wasn't true!");
/// # let a = 3; let b = 27;
/// debug_assert!(a + b == 30, "a = {}, b = {}", a, b);
/// ```
#[macro_export]
macro_rules! debug_assert(
    ($($arg:tt)*) => (if cfg!(not(ndebug)) { assert!($($arg)*); })
)

/// Asserts that two expressions are equal to each other, testing equality in
/// both directions.
///
/// On failure, this macro will print the values of the expressions.
///
/// Unlike `assert_eq!`, `debug_assert_eq!` statements can be disabled by
/// passing `--cfg ndebug` to the compiler. This makes `debug_assert_eq!`
/// useful for checks that are too expensive to be present in a release build
/// but may be helpful during development.
///
/// # Example
///
/// ```
/// let a = 3;
/// let b = 1 + 2;
/// debug_assert_eq!(a, b);
/// ```
#[macro_export]
macro_rules! debug_assert_eq(
    ($($arg:tt)*) => (if cfg!(not(ndebug)) { assert_eq!($($arg)*); })
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

/// Temporary transitionary thing.
#[macro_export]
macro_rules! format_strbuf(
    ($($arg:tt)*) => (
        format_args!(::std::fmt::format_strbuf, $($arg)*)
    )
)

/// Use the `format!` syntax to write data into a buffer of type `&mut Writer`.
/// See `std::fmt` for more information.
///
/// # Example
///
/// ```
/// # #![allow(unused_must_use)]
/// use std::io::MemWriter;
///
/// let mut w = MemWriter::new();
/// write!(&mut w, "test");
/// write!(&mut w, "formatted {}", "arguments");
/// ```
#[macro_export]
#[cfg(not(stage0))]
macro_rules! write(
    ($dst:expr, $($arg:tt)*) => ({
        format_args_method!($dst, write_fmt, $($arg)*)
    })
)
#[cfg(stage0)]
#[macro_export]
macro_rules! write(
    ($dst:expr, $($arg:tt)*) => ({
        format_args!(|args| { $dst.write_fmt(args) }, $($arg)*)
    })
)

/// Equivalent to the `write!` macro, except that a newline is appended after
/// the message is written.
#[macro_export]
macro_rules! writeln(
    ($dst:expr, $fmt:expr $($arg:tt)*) => (
        write!($dst, concat!($fmt, "\n") $($arg)*)
    )
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
/// local_data_key!(my_integer: int)
///
/// my_integer.replace(Some(2));
/// println!("{}", my_integer.get().map(|a| *a));
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

/// Create a `std::vec::Vec` containing the arguments.
#[macro_export]
macro_rules! vec(
    ($($e:expr),*) => ({
        // leading _ to allow empty construction without a warning.
        let mut _temp = ::std::vec::Vec::new();
        $(_temp.push($e);)*
        _temp
    });
    ($($e:expr),+,) => (vec!($($e),+))
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
/// let (tx1, rx1) = channel();
/// let (tx2, rx2) = channel();
/// # fn long_running_task() {}
/// # fn calculate_the_answer() -> int { 42 }
///
/// spawn(proc() { long_running_task(); tx1.send(()) });
/// spawn(proc() { tx2.send(calculate_the_answer()) });
///
/// select! (
///     () = rx1.recv() => println!("the long running task finished first"),
///     answer = rx2.recv() => {
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
        $($name:pat = $rx:ident.$meth:ident() => $code:expr),+
    ) => ({
        use std::comm::Select;
        let sel = Select::new();
        $( let mut $rx = sel.handle(&$rx); )+
        unsafe {
            $( $rx.add(); )+
        }
        let ret = sel.wait();
        $( if ret == $rx.id() { let $name = $rx.$meth(); $code } else )+
        { unreachable!() }
    })
}

// When testing the standard library, we link to the liblog crate to get the
// logging macros. In doing so, the liblog crate was linked against the real
// version of libstd, and uses a different std::fmt module than the test crate
// uses. To get around this difference, we redefine the log!() macro here to be
// just a dumb version of what it should be.
#[cfg(test)]
macro_rules! log (
    ($lvl:expr, $($args:tt)*) => (
        if log_enabled!($lvl) { println!($($args)*) }
    )
)

/// Built-in macros to the compiler itself.
///
/// These macros do not have any corresponding definition with a `macro_rules!`
/// macro, but are documented here. Their implementations can be found hardcoded
/// into libsyntax itself.
#[cfg(dox)]
pub mod builtin {
    /// The core macro for formatted string creation & output.
    ///
    /// This macro takes as its first argument a callable expression which will
    /// receive as its first argument a value of type `&fmt::Arguments`. This
    /// value can be passed to the functions in `std::fmt` for performing useful
    /// functions. All other formatting macros (`format!`, `write!`,
    /// `println!`, etc) are proxied through this one.
    ///
    /// For more information, see the documentation in `std::fmt`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::fmt;
    ///
    /// let s = format_args!(fmt::format, "hello {}", "world");
    /// assert_eq!(s, format!("hello {}", "world"));
    ///
    /// format_args!(|args| {
    ///     // pass `args` to another function, etc.
    /// }, "hello {}", "world");
    /// ```
    #[macro_export]
    macro_rules! format_args( ($closure:expr, $fmt:expr $($args:tt)*) => ({
        /* compiler built-in */
    }) )

    /// Inspect an environment variable at compile time.
    ///
    /// This macro will expand to the value of the named environment variable at
    /// compile time, yielding an expression of type `&'static str`.
    ///
    /// If the environment variable is not defined, then a compilation error
    /// will be emitted.  To not emit a compile error, use the `option_env!`
    /// macro instead.
    ///
    /// # Example
    ///
    /// ```rust
    /// let user: &'static str = env!("USER");
    /// println!("the user who compiled this code is: {}", user);
    /// ```
    #[macro_export]
    macro_rules! env( ($name:expr) => ({ /* compiler built-in */ }) )

    /// Optionally inspect an environment variable at compile time.
    ///
    /// If the named environment variable is present at compile time, this will
    /// expand into an expression of type `Option<&'static str>` whose value is
    /// `Some` of the value of the environment variable. If the environment
    /// variable is not present, then this will expand to `None`.
    ///
    /// A compile time error is never emitted when using this macro regardless
    /// of whether the environment variable is present or not.
    ///
    /// # Example
    ///
    /// ```rust
    /// let key: Option<&'static str> = option_env!("SECRET_KEY");
    /// println!("the secret key might be: {}", key);
    /// ```
    #[macro_export]
    macro_rules! option_env( ($name:expr) => ({ /* compiler built-in */ }) )

    /// Concatenate literals into a static byte slice.
    ///
    /// This macro takes any number of comma-separated literal expressions,
    /// yielding an expression of type `&'static [u8]` which is the
    /// concatenation (left to right) of all the literals in their byte format.
    ///
    /// This extension currently only supports string literals, character
    /// literals, and integers less than 256. The byte slice returned is the
    /// utf8-encoding of strings and characters.
    ///
    /// # Example
    ///
    /// ```
    /// let rust = bytes!("r", 'u', "st");
    /// assert_eq!(rust[1], 'u' as u8);
    /// ```
    #[macro_export]
    macro_rules! bytes( ($($e:expr),*) => ({ /* compiler built-in */ }) )

    /// Concatenate identifiers into one identifier.
    ///
    /// This macro takes any number of comma-separated identifiers, and
    /// concatenates them all into one, yielding an expression which is a new
    /// identifier. Note that hygiene makes it such that this macro cannot
    /// capture local variables, and macros are only allowed in item,
    /// statement or expression position, meaning this macro may be difficult to
    /// use in some situations.
    ///
    /// # Example
    ///
    /// ```
    /// fn foobar() -> int { 23 }
    ///
    /// let f = concat_idents!(foo, bar);
    /// println!("{}", f());
    /// ```
    #[macro_export]
    macro_rules! concat_idents( ($($e:ident),*) => ({ /* compiler built-in */ }) )

    /// Concatenates literals into a static string slice.
    ///
    /// This macro takes any number of comma-separated literals, yielding an
    /// expression of type `&'static str` which represents all of the literals
    /// concatenated left-to-right.
    ///
    /// Integer and floating point literals are stringified in order to be
    /// concatenated.
    ///
    /// # Example
    ///
    /// ```
    /// let s = concat!("test", 10, 'b', true);
    /// assert_eq!(s, "test10btrue");
    /// ```
    #[macro_export]
    macro_rules! concat( ($($e:expr),*) => ({ /* compiler built-in */ }) )

    /// A macro which expands to the line number on which it was invoked.
    ///
    /// The expanded expression has type `uint`, and the returned line is not
    /// the invocation of the `line!()` macro itself, but rather the first macro
    /// invocation leading up to the invocation of the `line!()` macro.
    ///
    /// # Example
    ///
    /// ```
    /// let current_line = line!();
    /// println!("defined on line: {}", current_line);
    /// ```
    #[macro_export]
    macro_rules! line( () => ({ /* compiler built-in */ }) )

    /// A macro which expands to the column number on which it was invoked.
    ///
    /// The expanded expression has type `uint`, and the returned column is not
    /// the invocation of the `col!()` macro itself, but rather the first macro
    /// invocation leading up to the invocation of the `col!()` macro.
    ///
    /// # Example
    ///
    /// ```
    /// let current_col = col!();
    /// println!("defined on column: {}", current_col);
    /// ```
    #[macro_export]
    macro_rules! col( () => ({ /* compiler built-in */ }) )

    /// A macro which expands to the file name from which it was invoked.
    ///
    /// The expanded expression has type `&'static str`, and the returned file
    /// is not the invocation of the `file!()` macro itself, but rather the
    /// first macro invocation leading up to the invocation of the `file!()`
    /// macro.
    ///
    /// # Example
    ///
    /// ```
    /// let this_file = file!();
    /// println!("defined in file: {}", this_file);
    /// ```
    #[macro_export]
    macro_rules! file( () => ({ /* compiler built-in */ }) )

    /// A macro which stringifies its argument.
    ///
    /// This macro will yield an expression of type `&'static str` which is the
    /// stringification of all the tokens passed to the macro. No restrictions
    /// are placed on the syntax of the macro invocation itself.
    ///
    /// # Example
    ///
    /// ```
    /// let one_plus_one = stringify!(1 + 1);
    /// assert_eq!(one_plus_one, "1 + 1");
    /// ```
    #[macro_export]
    macro_rules! stringify( ($t:tt) => ({ /* compiler built-in */ }) )

    /// Includes a utf8-encoded file as a string.
    ///
    /// This macro will yield an expression of type `&'static str` which is the
    /// contents of the filename specified. The file is located relative to the
    /// current file (similarly to how modules are found),
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let secret_key = include_str!("secret-key.ascii");
    /// ```
    #[macro_export]
    macro_rules! include_str( ($file:expr) => ({ /* compiler built-in */ }) )

    /// Includes a file as a byte slice.
    ///
    /// This macro will yield an expression of type `&'static [u8]` which is
    /// the contents of the filename specified. The file is located relative to
    /// the current file (similarly to how modules are found),
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let secret_key = include_bin!("secret-key.bin");
    /// ```
    #[macro_export]
    macro_rules! include_bin( ($file:expr) => ({ /* compiler built-in */ }) )

    /// Expands to a string that represents the current module path.
    ///
    /// The current module path can be thought of as the hierarchy of modules
    /// leading back up to the crate root. The first component of the path
    /// returned is the name of the crate currently being compiled.
    ///
    /// # Example
    ///
    /// ```rust
    /// mod test {
    ///     pub fn foo() {
    ///         assert!(module_path!().ends_with("test"));
    ///     }
    /// }
    ///
    /// test::foo();
    /// ```
    #[macro_export]
    macro_rules! module_path( () => ({ /* compiler built-in */ }) )

    /// Boolean evaluation of configuration flags.
    ///
    /// In addition to the `#[cfg]` attribute, this macro is provided to allow
    /// boolean expression evaluation of configuration flags. This frequently
    /// leads to less duplicated code.
    ///
    /// The syntax given to this macro is the same syntax as the `cfg`
    /// attribute.
    ///
    /// # Example
    ///
    /// ```rust
    /// let my_directory = if cfg!(windows) {
    ///     "windows-specific-directory"
    /// } else {
    ///     "unix-directory"
    /// };
    /// ```
    #[macro_export]
    macro_rules! cfg( ($cfg:tt) => ({ /* compiler built-in */ }) )
}
