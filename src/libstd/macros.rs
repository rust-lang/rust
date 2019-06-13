//! Standard library macros
//!
//! This modules contains a set of macros which are exported from the standard
//! library. Each macro is available for use when linking against the standard
//! library.

/// Panics the current thread.
///
/// This allows a program to terminate immediately and provide feedback
/// to the caller of the program. `panic!` should be used when a program reaches
/// an unrecoverable state.
///
/// This macro is the perfect way to assert conditions in example code and in
/// tests. `panic!` is closely tied with the `unwrap` method of both [`Option`]
/// and [`Result`][runwrap] enums. Both implementations call `panic!` when they are set
/// to None or Err variants.
///
/// This macro is used to inject panic into a Rust thread, causing the thread to
/// panic entirely. Each thread's panic can be reaped as the `Box<Any>` type,
/// and the single-argument form of the `panic!` macro will be the value which
/// is transmitted.
///
/// [`Result`] enum is often a better solution for recovering from errors than
/// using the `panic!` macro. This macro should be used to avoid proceeding using
/// incorrect values, such as from external sources. Detailed information about
/// error handling is found in the [book].
///
/// The multi-argument form of this macro panics with a string and has the
/// [`format!`] syntax for building a string.
///
/// See also the macro [`compile_error!`], for raising errors during compilation.
///
/// [runwrap]: ../std/result/enum.Result.html#method.unwrap
/// [`Option`]: ../std/option/enum.Option.html#method.unwrap
/// [`Result`]: ../std/result/enum.Result.html
/// [`format!`]: ../std/macro.format.html
/// [`compile_error!`]: ../std/macro.compile_error.html
/// [book]: ../book/ch09-00-error-handling.html
///
/// # Current implementation
///
/// If the main thread panics it will terminate all your threads and end your
/// program with code `101`.
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
#[allow_internal_unstable(__rust_unstable_column, libstd_sys_internals)]
macro_rules! panic {
    () => ({
        $crate::panic!("explicit panic")
    });
    ($msg:expr) => ({
        $crate::rt::begin_panic($msg, &(file!(), line!(), __rust_unstable_column!()))
    });
    ($msg:expr,) => ({
        $crate::panic!($msg)
    });
    ($fmt:expr, $($arg:tt)+) => ({
        $crate::rt::begin_panic_fmt(&format_args!($fmt, $($arg)+),
                                    &(file!(), line!(), __rust_unstable_column!()))
    });
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
/// Use `print!` only for the primary output of your program. Use
/// [`eprint!`] instead to print error and progress messages.
///
/// [`println!`]: ../std/macro.println.html
/// [flush]: ../std/io/trait.Write.html#tymethod.flush
/// [`eprint!`]: ../std/macro.eprint.html
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
#[allow_internal_unstable(print_internals)]
macro_rules! print {
    ($($arg:tt)*) => ($crate::io::_print(format_args!($($arg)*)));
}

/// Prints to the standard output, with a newline.
///
/// On all platforms, the newline is the LINE FEED character (`\n`/`U+000A`) alone
/// (no additional CARRIAGE RETURN (`\r`/`U+000D`).
///
/// Use the [`format!`] syntax to write data to the standard output.
/// See [`std::fmt`] for more information.
///
/// Use `println!` only for the primary output of your program. Use
/// [`eprintln!`] instead to print error and progress messages.
///
/// [`format!`]: ../std/macro.format.html
/// [`std::fmt`]: ../std/fmt/index.html
/// [`eprintln!`]: ../std/macro.eprintln.html
/// # Panics
///
/// Panics if writing to `io::stdout` fails.
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
#[allow_internal_unstable(print_internals, format_args_nl)]
macro_rules! println {
    () => ($crate::print!("\n"));
    ($($arg:tt)*) => ({
        $crate::io::_print(format_args_nl!($($arg)*));
    })
}

/// Prints to the standard error.
///
/// Equivalent to the [`print!`] macro, except that output goes to
/// [`io::stderr`] instead of `io::stdout`. See [`print!`] for
/// example usage.
///
/// Use `eprint!` only for error and progress messages. Use `print!`
/// instead for the primary output of your program.
///
/// [`io::stderr`]: ../std/io/struct.Stderr.html
/// [`print!`]: ../std/macro.print.html
///
/// # Panics
///
/// Panics if writing to `io::stderr` fails.
///
/// # Examples
///
/// ```
/// eprint!("Error: Could not complete task");
/// ```
#[macro_export]
#[stable(feature = "eprint", since = "1.19.0")]
#[allow_internal_unstable(print_internals)]
macro_rules! eprint {
    ($($arg:tt)*) => ($crate::io::_eprint(format_args!($($arg)*)));
}

/// Prints to the standard error, with a newline.
///
/// Equivalent to the [`println!`] macro, except that output goes to
/// [`io::stderr`] instead of `io::stdout`. See [`println!`] for
/// example usage.
///
/// Use `eprintln!` only for error and progress messages. Use `println!`
/// instead for the primary output of your program.
///
/// [`io::stderr`]: ../std/io/struct.Stderr.html
/// [`println!`]: ../std/macro.println.html
///
/// # Panics
///
/// Panics if writing to `io::stderr` fails.
///
/// # Examples
///
/// ```
/// eprintln!("Error: Could not complete task");
/// ```
#[macro_export]
#[stable(feature = "eprint", since = "1.19.0")]
#[allow_internal_unstable(print_internals, format_args_nl)]
macro_rules! eprintln {
    () => ($crate::eprint!("\n"));
    ($($arg:tt)*) => ({
        $crate::io::_eprint(format_args_nl!($($arg)*));
    })
}

/// Prints and returns the value of a given expression for quick and dirty
/// debugging.
///
/// An example:
///
/// ```rust
/// let a = 2;
/// let b = dbg!(a * 2) + 1;
/// //      ^-- prints: [src/main.rs:2] a * 2 = 4
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
/// should avoid having uses of it in version control for longer periods.
/// Use cases involving debug output that should be added to version control
/// are better served by macros such as [`debug!`] from the [`log`] crate.
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
/// [src/main.rs:4] n.checked_sub(4) = None
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
/// [src/main.rs:3] n <= 1 = false
/// [src/main.rs:3] n <= 1 = false
/// [src/main.rs:3] n <= 1 = false
/// [src/main.rs:3] n <= 1 = true
/// [src/main.rs:4] 1 = 1
/// [src/main.rs:5] n * factorial(n - 1) = 2
/// [src/main.rs:5] n * factorial(n - 1) = 6
/// [src/main.rs:5] n * factorial(n - 1) = 24
/// [src/main.rs:11] factorial(4) = 24
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
#[stable(feature = "dbg_macro", since = "1.32.0")]
macro_rules! dbg {
    () => {
        $crate::eprintln!("[{}:{}]", file!(), line!());
    };
    ($val:expr) => {
        // Use of `match` here is intentional because it affects the lifetimes
        // of temporaries - https://stackoverflow.com/a/48732525/1063961
        match $val {
            tmp => {
                $crate::eprintln!("[{}:{}] {} = {:#?}",
                    file!(), line!(), stringify!($val), &tmp);
                tmp
            }
        }
    };
    // Trailing comma with single argument is ignored
    ($val:expr,) => { $crate::dbg!($val) };
    ($($val:expr),+ $(,)?) => {
        ($($crate::dbg!($val)),+,)
    };
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
#[cfg(rustdoc)]
mod builtin {

    /// Causes compilation to fail with the given error message when encountered.
    ///
    /// This macro should be used when a crate uses a conditional compilation strategy to provide
    /// better error messages for erroneous conditions. It's the compiler-level form of [`panic!`],
    /// which emits an error at *runtime*, rather than during compilation.
    ///
    /// # Examples
    ///
    /// Two such examples are macros and `#[cfg]` environments.
    ///
    /// Emit better compiler error if a macro is passed invalid values. Without the final branch,
    /// the compiler would still emit an error, but the error's message would not mention the two
    /// valid values.
    ///
    /// ```compile_fail
    /// macro_rules! give_me_foo_or_bar {
    ///     (foo) => {};
    ///     (bar) => {};
    ///     ($x:ident) => {
    ///         compile_error!("This macro only accepts `foo` or `bar`");
    ///     }
    /// }
    ///
    /// give_me_foo_or_bar!(neither);
    /// // ^ will fail at compile time with message "This macro only accepts `foo` or `bar`"
    /// ```
    ///
    /// Emit compiler error if one of a number of features isn't available.
    ///
    /// ```compile_fail
    /// #[cfg(not(any(feature = "foo", feature = "bar")))]
    /// compile_error!("Either feature \"foo\" or \"bar\" must be enabled for this crate.");
    /// ```
    ///
    /// [`panic!`]: ../std/macro.panic.html
    #[stable(feature = "compile_error_macro", since = "1.20.0")]
    #[rustc_doc_only_macro]
    macro_rules! compile_error {
        ($msg:expr) => ({ /* compiler built-in */ });
        ($msg:expr,) => ({ /* compiler built-in */ });
    }

    /// Constructs parameters for the other string-formatting macros.
    ///
    /// This macro functions by taking a formatting string literal containing
    /// `{}` for each additional argument passed. `format_args!` prepares the
    /// additional parameters to ensure the output can be interpreted as a string
    /// and canonicalizes the arguments into a single type. Any value that implements
    /// the [`Display`] trait can be passed to `format_args!`, as can any
    /// [`Debug`] implementation be passed to a `{:?}` within the formatting string.
    ///
    /// This macro produces a value of type [`fmt::Arguments`]. This value can be
    /// passed to the macros within [`std::fmt`] for performing useful redirection.
    /// All other formatting macros ([`format!`], [`write!`], [`println!`], etc) are
    /// proxied through this one. `format_args!`, unlike its derived macros, avoids
    /// heap allocations.
    ///
    /// You can use the [`fmt::Arguments`] value that `format_args!` returns
    /// in `Debug` and `Display` contexts as seen below. The example also shows
    /// that `Debug` and `Display` format to the same thing: the interpolated
    /// format string in `format_args!`.
    ///
    /// ```rust
    /// let debug = format!("{:?}", format_args!("{} foo {:?}", 1, 2));
    /// let display = format!("{}", format_args!("{} foo {:?}", 1, 2));
    /// assert_eq!("1 foo 2", display);
    /// assert_eq!(display, debug);
    /// ```
    ///
    /// For more information, see the documentation in [`std::fmt`].
    ///
    /// [`Display`]: ../std/fmt/trait.Display.html
    /// [`Debug`]: ../std/fmt/trait.Debug.html
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
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_doc_only_macro]
    macro_rules! format_args {
        ($fmt:expr) => ({ /* compiler built-in */ });
        ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ });
    }

    /// Inspects an environment variable at compile time.
    ///
    /// This macro will expand to the value of the named environment variable at
    /// compile time, yielding an expression of type `&'static str`.
    ///
    /// If the environment variable is not defined, then a compilation error
    /// will be emitted. To not emit a compile error, use the [`option_env!`]
    /// macro instead.
    ///
    /// [`option_env!`]: ../std/macro.option_env.html
    ///
    /// # Examples
    ///
    /// ```
    /// let path: &'static str = env!("PATH");
    /// println!("the $PATH variable at the time of compiling was: {}", path);
    /// ```
    ///
    /// You can customize the error message by passing a string as the second
    /// parameter:
    ///
    /// ```compile_fail
    /// let doc: &'static str = env!("documentation", "what's that?!");
    /// ```
    ///
    /// If the `documentation` environment variable is not defined, you'll get
    /// the following error:
    ///
    /// ```text
    /// error: what's that?!
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_doc_only_macro]
    macro_rules! env {
        ($name:expr) => ({ /* compiler built-in */ });
        ($name:expr,) => ({ /* compiler built-in */ });
    }

    /// Optionally inspects an environment variable at compile time.
    ///
    /// If the named environment variable is present at compile time, this will
    /// expand into an expression of type `Option<&'static str>` whose value is
    /// `Some` of the value of the environment variable. If the environment
    /// variable is not present, then this will expand to `None`. See
    /// [`Option<T>`][option] for more information on this type.
    ///
    /// A compile time error is never emitted when using this macro regardless
    /// of whether the environment variable is present or not.
    ///
    /// [option]: ../std/option/enum.Option.html
    ///
    /// # Examples
    ///
    /// ```
    /// let key: Option<&'static str> = option_env!("SECRET_KEY");
    /// println!("the secret key might be: {:?}", key);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_doc_only_macro]
    macro_rules! option_env {
        ($name:expr) => ({ /* compiler built-in */ });
        ($name:expr,) => ({ /* compiler built-in */ });
    }

    /// Concatenates identifiers into one identifier.
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
    #[rustc_doc_only_macro]
    macro_rules! concat_idents {
        ($($e:ident),+) => ({ /* compiler built-in */ });
        ($($e:ident,)+) => ({ /* compiler built-in */ });
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
    #[rustc_doc_only_macro]
    macro_rules! concat {
        ($($e:expr),*) => ({ /* compiler built-in */ });
        ($($e:expr,)*) => ({ /* compiler built-in */ });
    }

    /// Expands to the line number on which it was invoked.
    ///
    /// With [`column!`] and [`file!`], these macros provide debugging information for
    /// developers about the location within the source.
    ///
    /// The expanded expression has type `u32` and is 1-based, so the first line
    /// in each file evaluates to 1, the second to 2, etc. This is consistent
    /// with error messages by common compilers or popular editors.
    /// The returned line is *not necessarily* the line of the `line!` invocation itself,
    /// but rather the first macro invocation leading up to the invocation
    /// of the `line!` macro.
    ///
    /// [`column!`]: macro.column.html
    /// [`file!`]: macro.file.html
    ///
    /// # Examples
    ///
    /// ```
    /// let current_line = line!();
    /// println!("defined on line: {}", current_line);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_doc_only_macro]
    macro_rules! line { () => ({ /* compiler built-in */ }) }

    /// Expands to the column number at which it was invoked.
    ///
    /// With [`line!`] and [`file!`], these macros provide debugging information for
    /// developers about the location within the source.
    ///
    /// The expanded expression has type `u32` and is 1-based, so the first column
    /// in each line evaluates to 1, the second to 2, etc. This is consistent
    /// with error messages by common compilers or popular editors.
    /// The returned column is *not necessarily* the line of the `column!` invocation itself,
    /// but rather the first macro invocation leading up to the invocation
    /// of the `column!` macro.
    ///
    /// [`line!`]: macro.line.html
    /// [`file!`]: macro.file.html
    ///
    /// # Examples
    ///
    /// ```
    /// let current_col = column!();
    /// println!("defined on column: {}", current_col);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_doc_only_macro]
    macro_rules! column { () => ({ /* compiler built-in */ }) }

    /// Expands to the file name in which it was invoked.
    ///
    /// With [`line!`] and [`column!`], these macros provide debugging information for
    /// developers about the location within the source.
    ///
    ///
    /// The expanded expression has type `&'static str`, and the returned file
    /// is not the invocation of the `file!` macro itself, but rather the
    /// first macro invocation leading up to the invocation of the `file!`
    /// macro.
    ///
    /// [`line!`]: macro.line.html
    /// [`column!`]: macro.column.html
    ///
    /// # Examples
    ///
    /// ```
    /// let this_file = file!();
    /// println!("defined in file: {}", this_file);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_doc_only_macro]
    macro_rules! file { () => ({ /* compiler built-in */ }) }

    /// Stringifies its arguments.
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
    #[rustc_doc_only_macro]
    macro_rules! stringify { ($($t:tt)*) => ({ /* compiler built-in */ }) }

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
    /// Assume there are two files in the same directory with the following
    /// contents:
    ///
    /// File 'spanish.in':
    ///
    /// ```text
    /// adiós
    /// ```
    ///
    /// File 'main.rs':
    ///
    /// ```ignore (cannot-doctest-external-file-dependency)
    /// fn main() {
    ///     let my_str = include_str!("spanish.in");
    ///     assert_eq!(my_str, "adiós\n");
    ///     print!("{}", my_str);
    /// }
    /// ```
    ///
    /// Compiling 'main.rs' and running the resulting binary will print "adiós".
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_doc_only_macro]
    macro_rules! include_str {
        ($file:expr) => ({ /* compiler built-in */ });
        ($file:expr,) => ({ /* compiler built-in */ });
    }

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
    /// Assume there are two files in the same directory with the following
    /// contents:
    ///
    /// File 'spanish.in':
    ///
    /// ```text
    /// adiós
    /// ```
    ///
    /// File 'main.rs':
    ///
    /// ```ignore (cannot-doctest-external-file-dependency)
    /// fn main() {
    ///     let bytes = include_bytes!("spanish.in");
    ///     assert_eq!(bytes, b"adi\xc3\xb3s\n");
    ///     print!("{}", String::from_utf8_lossy(bytes));
    /// }
    /// ```
    ///
    /// Compiling 'main.rs' and running the resulting binary will print "adiós".
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_doc_only_macro]
    macro_rules! include_bytes {
        ($file:expr) => ({ /* compiler built-in */ });
        ($file:expr,) => ({ /* compiler built-in */ });
    }

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
    #[rustc_doc_only_macro]
    macro_rules! module_path { () => ({ /* compiler built-in */ }) }

    /// Evaluates boolean combinations of configuration flags at compile-time.
    ///
    /// In addition to the `#[cfg]` attribute, this macro is provided to allow
    /// boolean expression evaluation of configuration flags. This frequently
    /// leads to less duplicated code.
    ///
    /// The syntax given to this macro is the same syntax as the [`cfg`]
    /// attribute.
    ///
    /// [`cfg`]: ../reference/conditional-compilation.html#the-cfg-attribute
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
    #[rustc_doc_only_macro]
    macro_rules! cfg { ($($cfg:tt)*) => ({ /* compiler built-in */ }) }

    /// Parses a file as an expression or an item according to the context.
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
    /// File 'monkeys.in':
    ///
    /// ```ignore (only-for-syntax-highlight)
    /// ['🙈', '🙊', '🙉']
    ///     .iter()
    ///     .cycle()
    ///     .take(6)
    ///     .collect::<String>()
    /// ```
    ///
    /// File 'main.rs':
    ///
    /// ```ignore (cannot-doctest-external-file-dependency)
    /// fn main() {
    ///     let my_string = include!("monkeys.in");
    ///     assert_eq!("🙈🙊🙉🙈🙊🙉", my_string);
    ///     println!("{}", my_string);
    /// }
    /// ```
    ///
    /// Compiling 'main.rs' and running the resulting binary will print
    /// "🙈🙊🙉🙈🙊🙉".
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_doc_only_macro]
    macro_rules! include {
        ($file:expr) => ({ /* compiler built-in */ });
        ($file:expr,) => ({ /* compiler built-in */ });
    }

    /// Asserts that a boolean expression is `true` at runtime.
    ///
    /// This will invoke the [`panic!`] macro if the provided expression cannot be
    /// evaluated to `true` at runtime.
    ///
    /// # Uses
    ///
    /// Assertions are always checked in both debug and release builds, and cannot
    /// be disabled. See [`debug_assert!`] for assertions that are not enabled in
    /// release builds by default.
    ///
    /// Unsafe code relies on `assert!` to enforce run-time invariants that, if
    /// violated could lead to unsafety.
    ///
    /// Other use-cases of `assert!` include testing and enforcing run-time
    /// invariants in safe code (whose violation cannot result in unsafety).
    ///
    /// # Custom Messages
    ///
    /// This macro has a second form, where a custom panic message can
    /// be provided with or without arguments for formatting. See [`std::fmt`]
    /// for syntax for this form.
    ///
    /// [`panic!`]: macro.panic.html
    /// [`debug_assert!`]: macro.debug_assert.html
    /// [`std::fmt`]: ../std/fmt/index.html
    ///
    /// # Examples
    ///
    /// ```
    /// // the panic message for these assertions is the stringified value of the
    /// // expression given.
    /// assert!(true);
    ///
    /// fn some_computation() -> bool { true } // a very simple function
    ///
    /// assert!(some_computation());
    ///
    /// // assert with a custom message
    /// let x = true;
    /// assert!(x, "x wasn't true!");
    ///
    /// let a = 3; let b = 27;
    /// assert!(a + b == 30, "a = {}, b = {}", a, b);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_doc_only_macro]
    macro_rules! assert {
        ($cond:expr) => ({ /* compiler built-in */ });
        ($cond:expr,) => ({ /* compiler built-in */ });
        ($cond:expr, $($arg:tt)+) => ({ /* compiler built-in */ });
    }
}
