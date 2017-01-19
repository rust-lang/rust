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

/// The entry point for panic of Rust threads.
///
/// This macro is used to inject panic into a Rust thread, causing the thread to
/// panic entirely. Each thread's panic can be reaped as the `Box<Any>` type,
/// and the single-argument form of the `panic!` macro will be the value which
/// is transmitted.
///
/// The multi-argument form of this macro panics with a string and has the
/// `format!` syntax for building a string.
///
/// # Examples
///
/// ```should_panic
/// # #![allow(unreachable_code)]
/// panic!();
/// panic!("this is a terrible mistake!");
/// panic!(4); // panic with the value of 4 to be collected elsewhere
/// panic!("this is a {} {message}", "fancy", message = "message");
/// ```
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
#[allow_internal_unstable]
macro_rules! panic {
    () => ({
        panic!("explicit panic")
    });
    ($msg:expr) => ({
        $crate::rt::begin_panic($msg, {
            // static requires less code at runtime, more constant data
            static _FILE_LINE: (&'static str, u32) = (file!(), line!());
            &_FILE_LINE
        })
    });
    ($fmt:expr, $($arg:tt)+) => ({
        $crate::rt::begin_panic_fmt(&format_args!($fmt, $($arg)+), {
            // The leading _'s are to avoid dead code warnings if this is
            // used inside a dead function. Just `#[allow(dead_code)]` is
            // insufficient, since the user may have
            // `#[forbid(dead_code)]` and which cannot be overridden.
            static _FILE_LINE: (&'static str, u32) = (file!(), line!());
            &_FILE_LINE
        })
    });
}

/// Macro for printing to the standard output.
///
/// Equivalent to the `println!` macro except that a newline is not printed at
/// the end of the message.
///
/// Note that stdout is frequently line-buffered by default so it may be
/// necessary to use `io::stdout().flush()` to ensure the output is emitted
/// immediately.
///
/// # Panics
///
/// Panics if writing to `io::stdout()` fails.
///
/// # Examples
///
/// ```
/// use std::io::{self, Write};
///
/// print!("this ");
/// print!("will ");
/// print!("be ");
/// print!("on ");
/// print!("the ");
/// print!("same ");
/// print!("line ");
///
/// io::stdout().flush().unwrap();
///
/// print!("this string has a newline, why not choose println! instead?\n");
///
/// io::stdout().flush().unwrap();
/// ```
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
#[allow_internal_unstable]
macro_rules! print {
    ($($arg:tt)*) => ($crate::io::_print(format_args!($($arg)*)));
}

/// Macro for printing to the standard output, with a newline. On all
/// platforms, the newline is the LINE FEED character (`\n`/`U+000A`) alone
/// (no additional CARRIAGE RETURN (`\r`/`U+000D`).
///
/// Use the `format!` syntax to write data to the standard output.
/// See `std::fmt` for more information.
///
/// # Panics
///
/// Panics if writing to `io::stdout()` fails.
///
/// # Examples
///
/// ```
/// println!(); // prints just a newline
/// println!("hello there!");
/// println!("format {} arguments", "some");
/// ```
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
macro_rules! println {
    () => (print!("\n"));
    ($fmt:expr) => (print!(concat!($fmt, "\n")));
    ($fmt:expr, $($arg:tt)*) => (print!(concat!($fmt, "\n"), $($arg)*));
}

/// A macro to select an event from a number of receivers.
///
/// This macro is used to wait for the first event to occur on a number of
/// receivers. It places no restrictions on the types of receivers given to
/// this macro, this can be viewed as a heterogeneous select.
///
/// # Examples
///
/// ```
/// #![feature(mpsc_select)]
///
/// use std::thread;
/// use std::sync::mpsc;
///
/// // two placeholder functions for now
/// fn long_running_thread() {}
/// fn calculate_the_answer() -> u32 { 42 }
///
/// let (tx1, rx1) = mpsc::channel();
/// let (tx2, rx2) = mpsc::channel();
///
/// thread::spawn(move|| { long_running_thread(); tx1.send(()).unwrap(); });
/// thread::spawn(move|| { tx2.send(calculate_the_answer()).unwrap(); });
///
/// select! {
///     _ = rx1.recv() => println!("the long running thread finished first"),
///     answer = rx2.recv() => {
///         println!("the answer was: {}", answer.unwrap());
///     }
/// }
/// # drop(rx1.recv());
/// # drop(rx2.recv());
/// ```
///
/// For more information about select, see the `std::sync::mpsc::Select` structure.
#[macro_export]
#[unstable(feature = "mpsc_select", issue = "27800")]
macro_rules! select {
    (
        $($name:pat = $rx:ident.$meth:ident() => $code:expr),+
    ) => ({
        use $crate::sync::mpsc::Select;
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

#[cfg(test)]
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => ({
        let (a, b) = (&$a, &$b);
        assert!((*a - *b).abs() < 1.0e-6,
                "{} is not approximately equal to {}", *a, *b);
    })
}

/// Built-in macros to the compiler itself.
///
/// These macros do not have any corresponding definition with a `macro_rules!`
/// macro, but are documented here. Their implementations can be found hardcoded
/// into libsyntax itself.
#[cfg(dox)]
pub mod builtin {
    /// The core macro for formatted string creation & output.
    ///
    /// This macro produces a value of type [`fmt::Arguments`]. This value can be
    /// passed to the functions in [`std::fmt`] for performing useful functions.
    /// All other formatting macros ([`format!`], [`write!`], [`println!`], etc) are
    /// proxied through this one.
    ///
    /// For more information, see the documentation in [`std::fmt`].
    ///
    /// [`fmt::Arguments`]: ../std/fmt/struct.Arguments.html
    /// [`std::fmt`]: ../std/fmt/index.html
    /// [`format!`]: ../std/macro.format.html
    /// [`write!`]: ../std/macro.write.html
    /// [`println!`]: ../std/macro.println.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt;
    ///
    /// let s = fmt::format(format_args!("hello {}", "world"));
    /// assert_eq!(s, format!("hello {}", "world"));
    ///
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[macro_export]
    macro_rules! format_args { ($fmt:expr, $($args:tt)*) => ({
        /* compiler built-in */
    }) }

    /// Inspect an environment variable at compile time.
    ///
    /// This macro will expand to the value of the named environment variable at
    /// compile time, yielding an expression of type `&'static str`.
    ///
    /// If the environment variable is not defined, then a compilation error
    /// will be emitted.  To not emit a compile error, use the `option_env!`
    /// macro instead.
    ///
    /// # Examples
    ///
    /// ```
    /// let path: &'static str = env!("PATH");
    /// println!("the $PATH variable at the time of compiling was: {}", path);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[macro_export]
    macro_rules! env { ($name:expr) => ({ /* compiler built-in */ }) }

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
    /// # Examples
    ///
    /// ```
    /// let key: Option<&'static str> = option_env!("SECRET_KEY");
    /// println!("the secret key might be: {:?}", key);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[macro_export]
    macro_rules! option_env { ($name:expr) => ({ /* compiler built-in */ }) }

    /// Concatenate identifiers into one identifier.
    ///
    /// This macro takes any number of comma-separated identifiers, and
    /// concatenates them all into one, yielding an expression which is a new
    /// identifier. Note that hygiene makes it such that this macro cannot
    /// capture local variables. Also, as a general rule, macros are only
    /// allowed in item, statement or expression position. That means while
    /// you may use this macro for referring to existing variables, functions or
    /// modules etc, you cannot define a new one with it.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(concat_idents)]
    ///
    /// # fn main() {
    /// fn foobar() -> u32 { 23 }
    ///
    /// let f = concat_idents!(foo, bar);
    /// println!("{}", f());
    ///
    /// // fn concat_idents!(new, fun, name) { } // not usable in this way!
    /// # }
    /// ```
    #[unstable(feature = "concat_idents_macro", issue = "29599")]
    #[macro_export]
    macro_rules! concat_idents {
        ($($e:ident),*) => ({ /* compiler built-in */ })
    }

    /// Concatenates literals into a static string slice.
    ///
    /// This macro takes any number of comma-separated literals, yielding an
    /// expression of type `&'static str` which represents all of the literals
    /// concatenated left-to-right.
    ///
    /// Integer and floating point literals are stringified in order to be
    /// concatenated.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = concat!("test", 10, 'b', true);
    /// assert_eq!(s, "test10btrue");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[macro_export]
    macro_rules! concat { ($($e:expr),*) => ({ /* compiler built-in */ }) }

    /// A macro which expands to the line number on which it was invoked.
    ///
    /// The expanded expression has type `u32`, and the returned line is not
    /// the invocation of the `line!()` macro itself, but rather the first macro
    /// invocation leading up to the invocation of the `line!()` macro.
    ///
    /// # Examples
    ///
    /// ```
    /// let current_line = line!();
    /// println!("defined on line: {}", current_line);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[macro_export]
    macro_rules! line { () => ({ /* compiler built-in */ }) }

    /// A macro which expands to the column number on which it was invoked.
    ///
    /// The expanded expression has type `u32`, and the returned column is not
    /// the invocation of the `column!()` macro itself, but rather the first macro
    /// invocation leading up to the invocation of the `column!()` macro.
    ///
    /// # Examples
    ///
    /// ```
    /// let current_col = column!();
    /// println!("defined on column: {}", current_col);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[macro_export]
    macro_rules! column { () => ({ /* compiler built-in */ }) }

    /// A macro which expands to the file name from which it was invoked.
    ///
    /// The expanded expression has type `&'static str`, and the returned file
    /// is not the invocation of the `file!()` macro itself, but rather the
    /// first macro invocation leading up to the invocation of the `file!()`
    /// macro.
    ///
    /// # Examples
    ///
    /// ```
    /// let this_file = file!();
    /// println!("defined in file: {}", this_file);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[macro_export]
    macro_rules! file { () => ({ /* compiler built-in */ }) }

    /// A macro which stringifies its argument.
    ///
    /// This macro will yield an expression of type `&'static str` which is the
    /// stringification of all the tokens passed to the macro. No restrictions
    /// are placed on the syntax of the macro invocation itself.
    ///
    /// Note that the expanded results of the input tokens may change in the
    /// future. You should be careful if you rely on the output.
    ///
    /// # Examples
    ///
    /// ```
    /// let one_plus_one = stringify!(1 + 1);
    /// assert_eq!(one_plus_one, "1 + 1");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[macro_export]
    macro_rules! stringify { ($t:tt) => ({ /* compiler built-in */ }) }

    /// Includes a utf8-encoded file as a string.
    ///
    /// The file is located relative to the current file. (similarly to how
    /// modules are found)
    ///
    /// This macro will yield an expression of type `&'static str` which is the
    /// contents of the file.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let secret_key = include_str!("secret-key.ascii");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[macro_export]
    macro_rules! include_str { ($file:expr) => ({ /* compiler built-in */ }) }

    /// Includes a file as a reference to a byte array.
    ///
    /// The file is located relative to the current file. (similarly to how
    /// modules are found)
    ///
    /// This macro will yield an expression of type `&'static [u8; N]` which is
    /// the contents of the file.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let secret_key = include_bytes!("secret-key.bin");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[macro_export]
    macro_rules! include_bytes { ($file:expr) => ({ /* compiler built-in */ }) }

    /// Expands to a string that represents the current module path.
    ///
    /// The current module path can be thought of as the hierarchy of modules
    /// leading back up to the crate root. The first component of the path
    /// returned is the name of the crate currently being compiled.
    ///
    /// # Examples
    ///
    /// ```
    /// mod test {
    ///     pub fn foo() {
    ///         assert!(module_path!().ends_with("test"));
    ///     }
    /// }
    ///
    /// test::foo();
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[macro_export]
    macro_rules! module_path { () => ({ /* compiler built-in */ }) }

    /// Boolean evaluation of configuration flags.
    ///
    /// In addition to the `#[cfg]` attribute, this macro is provided to allow
    /// boolean expression evaluation of configuration flags. This frequently
    /// leads to less duplicated code.
    ///
    /// The syntax given to this macro is the same syntax as [the `cfg`
    /// attribute](../reference.html#conditional-compilation).
    ///
    /// # Examples
    ///
    /// ```
    /// let my_directory = if cfg!(windows) {
    ///     "windows-specific-directory"
    /// } else {
    ///     "unix-directory"
    /// };
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[macro_export]
    macro_rules! cfg { ($($cfg:tt)*) => ({ /* compiler built-in */ }) }

    /// Parse a file as an expression or an item according to the context.
    ///
    /// The file is located relative to the current file (similarly to how
    /// modules are found).
    ///
    /// Using this macro is often a bad idea, because if the file is
    /// parsed as an expression, it is going to be placed in the
    /// surrounding code unhygienically. This could result in variables
    /// or functions being different from what the file expected if
    /// there are variables or functions that have the same name in
    /// the current file.
    ///
    /// # Examples
    ///
    /// Assume there are two files in the same directory with the following
    /// contents:
    ///
    /// File 'my_str.in':
    ///
    /// ```ignore
    /// "Hello World!"
    /// ```
    ///
    /// File 'main.rs':
    ///
    /// ```ignore
    /// fn main() {
    ///     let my_str = include!("my_str.in");
    ///     println!("{}", my_str);
    /// }
    /// ```
    ///
    /// Compiling 'main.rs' and running the resulting binary will print "Hello
    /// World!".
    #[stable(feature = "rust1", since = "1.0.0")]
    #[macro_export]
    macro_rules! include { ($file:expr) => ({ /* compiler built-in */ }) }
}
