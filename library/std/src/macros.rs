//! Standard library macros
//!
//! This module contains a set of macros which are exported from the standard
//! library. Each macro is available for use when linking against the standard
//! library.
// ignore-tidy-dbg

#[doc = include_str!("../../core/src/macros/panic.md")]
#[macro_export]
#[rustc_builtin_macro(std_panic)]
#[stable(feature = "rust1", since = "1.0.0")]
#[allow_internal_unstable(edition_panic)]
#[cfg_attr(not(test), rustc_diagnostic_item = "std_panic_macro")]
macro_rules! panic {
    // Expands to either `$crate::panic::panic_2015` or `$crate::panic::panic_2021`
    // depending on the edition of the caller.
    ($($arg:tt)*) => {
        /* compiler built-in */
    };
}

/// Prints to the standard output.
///
/// Equivalent to the [`println!`] macro except that a newline is not printed at
/// the end of the message.
///
/// Note that stdout is frequently line-buffered by default so it may be
/// necessary to use [`io::stdout().flush()`][flush] to ensure the output is emitted
/// immediately.
///
/// The `print!` macro will lock the standard output on each call. If you call
/// `print!` within a hot loop, this behavior may be the bottleneck of the loop.
/// To avoid this, lock stdout with [`io::stdout().lock()`][lock]:
/// ```
/// use std::io::{stdout, Write};
///
/// let mut lock = stdout().lock();
/// write!(lock, "hello world").unwrap();
/// ```
///
/// Use `print!` only for the primary output of your program. Use
/// [`eprint!`] instead to print error and progress messages.
///
/// See the formatting documentation in [`std::fmt`](crate::fmt)
/// for details of the macro argument syntax.
///
/// [flush]: crate::io::Write::flush
/// [`println!`]: crate::println
/// [`eprint!`]: crate::eprint
/// [lock]: crate::io::Stdout
///
/// # Panics
///
/// Panics if writing to `io::stdout()` fails.
///
/// Writing to non-blocking stdout can cause an error, which will lead
/// this macro to panic.
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
#[cfg_attr(not(test), rustc_diagnostic_item = "print_macro")]
#[allow_internal_unstable(print_internals)]
macro_rules! print {
    ($($arg:tt)*) => {{
        $crate::io::_print($crate::format_args!($($arg)*));
    }};
}

/// Prints to the standard output, with a newline.
///
/// On all platforms, the newline is the LINE FEED character (`\n`/`U+000A`) alone
/// (no additional CARRIAGE RETURN (`\r`/`U+000D`)).
///
/// This macro uses the same syntax as [`format!`], but writes to the standard output instead.
/// See [`std::fmt`] for more information.
///
/// The `println!` macro will lock the standard output on each call. If you call
/// `println!` within a hot loop, this behavior may be the bottleneck of the loop.
/// To avoid this, lock stdout with [`io::stdout().lock()`][lock]:
/// ```
/// use std::io::{stdout, Write};
///
/// let mut lock = stdout().lock();
/// writeln!(lock, "hello world").unwrap();
/// ```
///
/// Use `println!` only for the primary output of your program. Use
/// [`eprintln!`] instead to print error and progress messages.
///
/// See the formatting documentation in [`std::fmt`](crate::fmt)
/// for details of the macro argument syntax.
///
/// [`std::fmt`]: crate::fmt
/// [`eprintln!`]: crate::eprintln
/// [lock]: crate::io::Stdout
///
/// # Panics
///
/// Panics if writing to [`io::stdout`] fails.
///
/// Writing to non-blocking stdout can cause an error, which will lead
/// this macro to panic.
///
/// [`io::stdout`]: crate::io::stdout
///
/// # Examples
///
/// ```
/// println!(); // prints just a newline
/// println!("hello there!");
/// println!("format {} arguments", "some");
/// let local_variable = "some";
/// println!("format {local_variable} arguments");
/// ```
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "println_macro")]
#[allow_internal_unstable(print_internals, format_args_nl)]
macro_rules! println {
    () => {
        $crate::print!("\n")
    };
    ($($arg:tt)*) => {{
        $crate::io::_print($crate::format_args_nl!($($arg)*));
    }};
}

/// Prints to the standard error.
///
/// Equivalent to the [`print!`] macro, except that output goes to
/// [`io::stderr`] instead of [`io::stdout`]. See [`print!`] for
/// example usage.
///
/// Use `eprint!` only for error and progress messages. Use `print!`
/// instead for the primary output of your program.
///
/// [`io::stderr`]: crate::io::stderr
/// [`io::stdout`]: crate::io::stdout
///
/// See the formatting documentation in [`std::fmt`](crate::fmt)
/// for details of the macro argument syntax.
///
/// # Panics
///
/// Panics if writing to `io::stderr` fails.
///
/// Writing to non-blocking stderr can cause an error, which will lead
/// this macro to panic.
///
/// # Examples
///
/// ```
/// eprint!("Error: Could not complete task");
/// ```
#[macro_export]
#[stable(feature = "eprint", since = "1.19.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "eprint_macro")]
#[allow_internal_unstable(print_internals)]
macro_rules! eprint {
    ($($arg:tt)*) => {{
        $crate::io::_eprint($crate::format_args!($($arg)*));
    }};
}

/// Prints to the standard error, with a newline.
///
/// Equivalent to the [`println!`] macro, except that output goes to
/// [`io::stderr`] instead of [`io::stdout`]. See [`println!`] for
/// example usage.
///
/// Use `eprintln!` only for error and progress messages. Use `println!`
/// instead for the primary output of your program.
///
/// See the formatting documentation in [`std::fmt`](crate::fmt)
/// for details of the macro argument syntax.
///
/// [`io::stderr`]: crate::io::stderr
/// [`io::stdout`]: crate::io::stdout
/// [`println!`]: crate::println
///
/// # Panics
///
/// Panics if writing to `io::stderr` fails.
///
/// Writing to non-blocking stderr can cause an error, which will lead
/// this macro to panic.
///
/// # Examples
///
/// ```
/// eprintln!("Error: Could not complete task");
/// ```
#[macro_export]
#[stable(feature = "eprint", since = "1.19.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "eprintln_macro")]
#[allow_internal_unstable(print_internals, format_args_nl)]
macro_rules! eprintln {
    () => {
        $crate::eprint!("\n")
    };
    ($($arg:tt)*) => {{
        $crate::io::_eprint($crate::format_args_nl!($($arg)*));
    }};
}

/// Prints and returns the value of a given expression for quick and dirty
/// debugging.
///
/// An example:
///
/// ```rust
/// let a = 2;
/// let b = dbg!(a * 2) + 1;
/// //      ^-- prints: [src/main.rs:2:9] a * 2 = 4
/// assert_eq!(b, 5);
/// ```
///
/// The macro works by using the `Debug` implementation of the type of
/// the given expression to print the value to [stderr] along with the
/// source location of the macro invocation as well as the source code
/// of the expression.
///
/// Invoking the macro on an expression moves and takes ownership of it
/// before returning the evaluated expression unchanged. If the type
/// of the expression does not implement `Copy` and you don't want
/// to give up ownership, you can instead borrow with `dbg!(&expr)`
/// for some expression `expr`.
///
/// The `dbg!` macro works exactly the same in release builds.
/// This is useful when debugging issues that only occur in release
/// builds or when debugging in release mode is significantly faster.
///
/// Note that the macro is intended as a debugging tool and therefore you
/// should avoid having uses of it in version control for long periods
/// (other than in tests and similar).
/// Debug output from production code is better done with other facilities
/// such as the [`debug!`] macro from the [`log`] crate.
///
/// # Stability
///
/// The exact output printed by this macro should not be relied upon
/// and is subject to future changes.
///
/// # Panics
///
/// Panics if writing to `io::stderr` fails.
///
/// # Further examples
///
/// With a method call:
///
/// ```rust
/// fn foo(n: usize) {
///     if let Some(_) = dbg!(n.checked_sub(4)) {
///         // ...
///     }
/// }
///
/// foo(3)
/// ```
///
/// This prints to [stderr]:
///
/// ```text,ignore
/// [src/main.rs:2:22] n.checked_sub(4) = None
/// ```
///
/// Naive factorial implementation:
///
/// ```rust
/// fn factorial(n: u32) -> u32 {
///     if dbg!(n <= 1) {
///         dbg!(1)
///     } else {
///         dbg!(n * factorial(n - 1))
///     }
/// }
///
/// dbg!(factorial(4));
/// ```
///
/// This prints to [stderr]:
///
/// ```text,ignore
/// [src/main.rs:2:8] n <= 1 = false
/// [src/main.rs:2:8] n <= 1 = false
/// [src/main.rs:2:8] n <= 1 = false
/// [src/main.rs:2:8] n <= 1 = true
/// [src/main.rs:3:9] 1 = 1
/// [src/main.rs:7:9] n * factorial(n - 1) = 2
/// [src/main.rs:7:9] n * factorial(n - 1) = 6
/// [src/main.rs:7:9] n * factorial(n - 1) = 24
/// [src/main.rs:9:1] factorial(4) = 24
/// ```
///
/// The `dbg!(..)` macro moves the input:
///
/// ```compile_fail
/// /// A wrapper around `usize` which importantly is not Copyable.
/// #[derive(Debug)]
/// struct NoCopy(usize);
///
/// let a = NoCopy(42);
/// let _ = dbg!(a); // <-- `a` is moved here.
/// let _ = dbg!(a); // <-- `a` is moved again; error!
/// ```
///
/// You can also use `dbg!()` without a value to just print the
/// file and line whenever it's reached.
///
/// Finally, if you want to `dbg!(..)` multiple values, it will treat them as
/// a tuple (and return it, too):
///
/// ```
/// assert_eq!(dbg!(1usize, 2u32), (1, 2));
/// ```
///
/// However, a single argument with a trailing comma will still not be treated
/// as a tuple, following the convention of ignoring trailing commas in macro
/// invocations. You can use a 1-tuple directly if you need one:
///
/// ```
/// assert_eq!(1, dbg!(1u32,)); // trailing comma ignored
/// assert_eq!((1,), dbg!((1u32,))); // 1-tuple
/// ```
///
/// [stderr]: https://en.wikipedia.org/wiki/Standard_streams#Standard_error_(stderr)
/// [`debug!`]: https://docs.rs/log/*/log/macro.debug.html
/// [`log`]: https://crates.io/crates/log
#[macro_export]
#[allow_internal_unstable(std_internals)]
#[cfg_attr(not(test), rustc_diagnostic_item = "dbg_macro")]
#[stable(feature = "dbg_macro", since = "1.32.0")]
macro_rules! dbg {
    () => {
        $crate::eprintln!("[{}:{}:{}]", $crate::file!(), $crate::line!(), $crate::column!())
    };
    ($($val:expr),+ $(,)?) => {
        $crate::macros::dbg_internal!(() () ($($val),+))
    };
}

/// Internal macro that processes a list of expressions and produces a chain of
/// nested `match`es, one for each expression, before finally calling `eprint!`
/// with the collected information and returning all the evaluated expressions
/// in a tuple.
///
/// E.g. `dbg_internal!(() () (1, 2))` expands into
/// ```rust, ignore
/// match 1 {
///     tmp_1 => match 2 {
///         tmp_2 => {
///             eprint!("...", &tmp_1, &tmp_2, /* some other arguments */);
///             (tmp_1, tmp_2)
///         }
///     }
/// }
/// ```
///
/// This is necessary so that `dbg!` outputs don't get torn, see #136703.
#[doc(hidden)]
#[rustc_macro_transparency = "semiopaque"]
pub macro dbg_internal {
    (($($piece:literal),+) ($($processed:expr => $bound:expr),+) ()) => {{
        $crate::eprint!(
            $crate::concat!($($piece),+),
            $(
                $crate::stringify!($processed),
                // The `&T: Debug` check happens here (not in the format literal desugaring)
                // to avoid format literal related messages and suggestions.
                &&$bound as &dyn $crate::fmt::Debug
            ),+,
            // The location returned here is that of the macro invocation, so
            // it will be the same for all expressions. Thus, label these
            // arguments so that they can be reused in every piece of the
            // formatting template.
            file=$crate::file!(),
            line=$crate::line!(),
            column=$crate::column!()
        );
        // Comma separate the variables only when necessary so that this will
        // not yield a tuple for a single expression, but rather just parenthesize
        // the expression.
        ($($bound),+)
    }},
    (($($piece:literal),*) ($($processed:expr => $bound:expr),*) ($val:expr $(,$rest:expr)*)) => {
        // Use of `match` here is intentional because it affects the lifetimes
        // of temporaries - https://stackoverflow.com/a/48732525/1063961
        match $val {
            tmp => $crate::macros::dbg_internal!(
                ($($piece,)* "[{file}:{line}:{column}] {} = {:#?}\n")
                ($($processed => $bound,)* $val => tmp)
                ($($rest),*)
            ),
        }
    },
}
