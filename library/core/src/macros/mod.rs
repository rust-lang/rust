#[doc = include_str!("panic.md")]
#[macro_export]
#[rustc_builtin_macro(core_panic)]
#[allow_internal_unstable(edition_panic)]
#[stable(feature = "core", since = "1.6.0")]
#[rustc_diagnostic_item = "core_panic_macro"]
macro_rules! panic {
    // Expands to either `$crate::panic::panic_2015` or `$crate::panic::panic_2021`
    // depending on the edition of the caller.
    ($($arg:tt)*) => {
        /* compiler built-in */
    };
}

/// Asserts that two expressions are equal to each other (using [`PartialEq`]).
///
/// On panic, this macro will print the values of the expressions with their
/// debug representations.
///
/// Like [`assert!`], this macro has a second form, where a custom
/// panic message can be provided.
///
/// # Examples
///
/// ```
/// let a = 3;
/// let b = 1 + 2;
/// assert_eq!(a, b);
///
/// assert_eq!(a, b, "we are testing addition with {} and {}", a, b);
/// ```
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
#[allow_internal_unstable(core_panic)]
macro_rules! assert_eq {
    ($left:expr, $right:expr $(,)?) => ({
        match (&$left, &$right) {
            (left_val, right_val) => {
                if !(*left_val == *right_val) {
                    let kind = $crate::panicking::AssertKind::Eq;
                    // The reborrows below are intentional. Without them, the stack slot for the
                    // borrow is initialized even before the values are compared, leading to a
                    // noticeable slow down.
                    $crate::panicking::assert_failed(kind, &*left_val, &*right_val, $crate::option::Option::None);
                }
            }
        }
    });
    ($left:expr, $right:expr, $($arg:tt)+) => ({
        match (&$left, &$right) {
            (left_val, right_val) => {
                if !(*left_val == *right_val) {
                    let kind = $crate::panicking::AssertKind::Eq;
                    // The reborrows below are intentional. Without them, the stack slot for the
                    // borrow is initialized even before the values are compared, leading to a
                    // noticeable slow down.
                    $crate::panicking::assert_failed(kind, &*left_val, &*right_val, $crate::option::Option::Some($crate::format_args!($($arg)+)));
                }
            }
        }
    });
}

/// Asserts that two expressions are not equal to each other (using [`PartialEq`]).
///
/// On panic, this macro will print the values of the expressions with their
/// debug representations.
///
/// Like [`assert!`], this macro has a second form, where a custom
/// panic message can be provided.
///
/// # Examples
///
/// ```
/// let a = 3;
/// let b = 2;
/// assert_ne!(a, b);
///
/// assert_ne!(a, b, "we are testing that the values are not equal");
/// ```
#[macro_export]
#[stable(feature = "assert_ne", since = "1.13.0")]
#[allow_internal_unstable(core_panic)]
macro_rules! assert_ne {
    ($left:expr, $right:expr $(,)?) => ({
        match (&$left, &$right) {
            (left_val, right_val) => {
                if *left_val == *right_val {
                    let kind = $crate::panicking::AssertKind::Ne;
                    // The reborrows below are intentional. Without them, the stack slot for the
                    // borrow is initialized even before the values are compared, leading to a
                    // noticeable slow down.
                    $crate::panicking::assert_failed(kind, &*left_val, &*right_val, $crate::option::Option::None);
                }
            }
        }
    });
    ($left:expr, $right:expr, $($arg:tt)+) => ({
        match (&($left), &($right)) {
            (left_val, right_val) => {
                if *left_val == *right_val {
                    let kind = $crate::panicking::AssertKind::Ne;
                    // The reborrows below are intentional. Without them, the stack slot for the
                    // borrow is initialized even before the values are compared, leading to a
                    // noticeable slow down.
                    $crate::panicking::assert_failed(kind, &*left_val, &*right_val, $crate::option::Option::Some($crate::format_args!($($arg)+)));
                }
            }
        }
    });
}

/// Asserts that an expression matches any of the given patterns.
///
/// Like in a `match` expression, the pattern can be optionally followed by `if`
/// and a guard expression that has access to names bound by the pattern.
///
/// On panic, this macro will print the value of the expression with its
/// debug representation.
///
/// Like [`assert!`], this macro has a second form, where a custom
/// panic message can be provided.
///
/// # Examples
///
/// ```
/// #![feature(assert_matches)]
///
/// use std::assert_matches::assert_matches;
///
/// let a = 1u32.checked_add(2);
/// let b = 1u32.checked_sub(2);
/// assert_matches!(a, Some(_));
/// assert_matches!(b, None);
///
/// let c = Ok("abc".to_string());
/// assert_matches!(c, Ok(x) | Err(x) if x.len() < 100);
/// ```
#[unstable(feature = "assert_matches", issue = "82775")]
#[allow_internal_unstable(core_panic)]
#[rustc_macro_transparency = "semitransparent"]
pub macro assert_matches {
    ($left:expr, $(|)? $( $pattern:pat_param )|+ $( if $guard: expr )? $(,)?) => ({
        match $left {
            $( $pattern )|+ $( if $guard )? => {}
            ref left_val => {
                $crate::panicking::assert_matches_failed(
                    left_val,
                    $crate::stringify!($($pattern)|+ $(if $guard)?),
                    $crate::option::Option::None
                );
            }
        }
    }),
    ($left:expr, $(|)? $( $pattern:pat_param )|+ $( if $guard: expr )?, $($arg:tt)+) => ({
        match $left {
            $( $pattern )|+ $( if $guard )? => {}
            ref left_val => {
                $crate::panicking::assert_matches_failed(
                    left_val,
                    $crate::stringify!($($pattern)|+ $(if $guard)?),
                    $crate::option::Option::Some($crate::format_args!($($arg)+))
                );
            }
        }
    }),
}

/// Asserts that a boolean expression is `true` at runtime.
///
/// This will invoke the [`panic!`] macro if the provided expression cannot be
/// evaluated to `true` at runtime.
///
/// Like [`assert!`], this macro also has a second version, where a custom panic
/// message can be provided.
///
/// # Uses
///
/// Unlike [`assert!`], `debug_assert!` statements are only enabled in non
/// optimized builds by default. An optimized build will not execute
/// `debug_assert!` statements unless `-C debug-assertions` is passed to the
/// compiler. This makes `debug_assert!` useful for checks that are too
/// expensive to be present in a release build but may be helpful during
/// development. The result of expanding `debug_assert!` is always type checked.
///
/// An unchecked assertion allows a program in an inconsistent state to keep
/// running, which might have unexpected consequences but does not introduce
/// unsafety as long as this only happens in safe code. The performance cost
/// of assertions, however, is not measurable in general. Replacing [`assert!`]
/// with `debug_assert!` is thus only encouraged after thorough profiling, and
/// more importantly, only in safe code!
///
/// # Examples
///
/// ```
/// // the panic message for these assertions is the stringified value of the
/// // expression given.
/// debug_assert!(true);
///
/// fn some_expensive_computation() -> bool { true } // a very simple function
/// debug_assert!(some_expensive_computation());
///
/// // assert with a custom message
/// let x = true;
/// debug_assert!(x, "x wasn't true!");
///
/// let a = 3; let b = 27;
/// debug_assert!(a + b == 30, "a = {}, b = {}", a, b);
/// ```
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_diagnostic_item = "debug_assert_macro"]
#[allow_internal_unstable(edition_panic)]
macro_rules! debug_assert {
    ($($arg:tt)*) => (if $crate::cfg!(debug_assertions) { $crate::assert!($($arg)*); })
}

/// Asserts that two expressions are equal to each other.
///
/// On panic, this macro will print the values of the expressions with their
/// debug representations.
///
/// Unlike [`assert_eq!`], `debug_assert_eq!` statements are only enabled in non
/// optimized builds by default. An optimized build will not execute
/// `debug_assert_eq!` statements unless `-C debug-assertions` is passed to the
/// compiler. This makes `debug_assert_eq!` useful for checks that are too
/// expensive to be present in a release build but may be helpful during
/// development. The result of expanding `debug_assert_eq!` is always type checked.
///
/// # Examples
///
/// ```
/// let a = 3;
/// let b = 1 + 2;
/// debug_assert_eq!(a, b);
/// ```
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
macro_rules! debug_assert_eq {
    ($($arg:tt)*) => (if $crate::cfg!(debug_assertions) { $crate::assert_eq!($($arg)*); })
}

/// Asserts that two expressions are not equal to each other.
///
/// On panic, this macro will print the values of the expressions with their
/// debug representations.
///
/// Unlike [`assert_ne!`], `debug_assert_ne!` statements are only enabled in non
/// optimized builds by default. An optimized build will not execute
/// `debug_assert_ne!` statements unless `-C debug-assertions` is passed to the
/// compiler. This makes `debug_assert_ne!` useful for checks that are too
/// expensive to be present in a release build but may be helpful during
/// development. The result of expanding `debug_assert_ne!` is always type checked.
///
/// # Examples
///
/// ```
/// let a = 3;
/// let b = 2;
/// debug_assert_ne!(a, b);
/// ```
#[macro_export]
#[stable(feature = "assert_ne", since = "1.13.0")]
macro_rules! debug_assert_ne {
    ($($arg:tt)*) => (if $crate::cfg!(debug_assertions) { $crate::assert_ne!($($arg)*); })
}

/// Asserts that an expression matches any of the given patterns.
///
/// Like in a `match` expression, the pattern can be optionally followed by `if`
/// and a guard expression that has access to names bound by the pattern.
///
/// On panic, this macro will print the value of the expression with its
/// debug representation.
///
/// Unlike [`assert_matches!`], `debug_assert_matches!` statements are only
/// enabled in non optimized builds by default. An optimized build will not
/// execute `debug_assert_matches!` statements unless `-C debug-assertions` is
/// passed to the compiler. This makes `debug_assert_matches!` useful for
/// checks that are too expensive to be present in a release build but may be
/// helpful during development. The result of expanding `debug_assert_matches!`
/// is always type checked.
///
/// # Examples
///
/// ```
/// #![feature(assert_matches)]
///
/// use std::assert_matches::debug_assert_matches;
///
/// let a = 1u32.checked_add(2);
/// let b = 1u32.checked_sub(2);
/// debug_assert_matches!(a, Some(_));
/// debug_assert_matches!(b, None);
///
/// let c = Ok("abc".to_string());
/// debug_assert_matches!(c, Ok(x) | Err(x) if x.len() < 100);
/// ```
#[macro_export]
#[unstable(feature = "assert_matches", issue = "82775")]
#[allow_internal_unstable(assert_matches)]
#[rustc_macro_transparency = "semitransparent"]
pub macro debug_assert_matches($($arg:tt)*) {
    if $crate::cfg!(debug_assertions) { $crate::assert_matches::assert_matches!($($arg)*); }
}

/// Returns whether the given expression matches any of the given patterns.
///
/// Like in a `match` expression, the pattern can be optionally followed by `if`
/// and a guard expression that has access to names bound by the pattern.
///
/// # Examples
///
/// ```
/// let foo = 'f';
/// assert!(matches!(foo, 'A'..='Z' | 'a'..='z'));
///
/// let bar = Some(4);
/// assert!(matches!(bar, Some(x) if x > 2));
/// ```
#[macro_export]
#[stable(feature = "matches_macro", since = "1.42.0")]
macro_rules! matches {
    ($expression:expr, $(|)? $( $pattern:pat_param )|+ $( if $guard: expr )? $(,)?) => {
        match $expression {
            $( $pattern )|+ $( if $guard )? => true,
            _ => false
        }
    }
}

/// Unwraps a result or propagates its error.
///
/// The `?` operator was added to replace `try!` and should be used instead.
/// Furthermore, `try` is a reserved word in Rust 2018, so if you must use
/// it, you will need to use the [raw-identifier syntax][ris]: `r#try`.
///
/// [ris]: https://doc.rust-lang.org/nightly/rust-by-example/compatibility/raw_identifiers.html
///
/// `try!` matches the given [`Result`]. In case of the `Ok` variant, the
/// expression has the value of the wrapped value.
///
/// In case of the `Err` variant, it retrieves the inner error. `try!` then
/// performs conversion using `From`. This provides automatic conversion
/// between specialized errors and more general ones. The resulting
/// error is then immediately returned.
///
/// Because of the early return, `try!` can only be used in functions that
/// return [`Result`].
///
/// # Examples
///
/// ```
/// use std::io;
/// use std::fs::File;
/// use std::io::prelude::*;
///
/// enum MyError {
///     FileWriteError
/// }
///
/// impl From<io::Error> for MyError {
///     fn from(e: io::Error) -> MyError {
///         MyError::FileWriteError
///     }
/// }
///
/// // The preferred method of quick returning Errors
/// fn write_to_file_question() -> Result<(), MyError> {
///     let mut file = File::create("my_best_friends.txt")?;
///     file.write_all(b"This is a list of my best friends.")?;
///     Ok(())
/// }
///
/// // The previous method of quick returning Errors
/// fn write_to_file_using_try() -> Result<(), MyError> {
///     let mut file = r#try!(File::create("my_best_friends.txt"));
///     r#try!(file.write_all(b"This is a list of my best friends."));
///     Ok(())
/// }
///
/// // This is equivalent to:
/// fn write_to_file_using_match() -> Result<(), MyError> {
///     let mut file = r#try!(File::create("my_best_friends.txt"));
///     match file.write_all(b"This is a list of my best friends.") {
///         Ok(v) => v,
///         Err(e) => return Err(From::from(e)),
///     }
///     Ok(())
/// }
/// ```
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_deprecated(since = "1.39.0", reason = "use the `?` operator instead")]
#[doc(alias = "?")]
macro_rules! r#try {
    ($expr:expr $(,)?) => {
        match $expr {
            $crate::result::Result::Ok(val) => val,
            $crate::result::Result::Err(err) => {
                return $crate::result::Result::Err($crate::convert::From::from(err));
            }
        }
    };
}

/// Writes formatted data into a buffer.
///
/// This macro accepts a 'writer', a format string, and a list of arguments. Arguments will be
/// formatted according to the specified format string and the result will be passed to the writer.
/// The writer may be any value with a `write_fmt` method; generally this comes from an
/// implementation of either the [`fmt::Write`] or the [`io::Write`] trait. The macro
/// returns whatever the `write_fmt` method returns; commonly a [`fmt::Result`], or an
/// [`io::Result`].
///
/// See [`std::fmt`] for more information on the format string syntax.
///
/// [`std::fmt`]: ../std/fmt/index.html
/// [`fmt::Write`]: crate::fmt::Write
/// [`io::Write`]: ../std/io/trait.Write.html
/// [`fmt::Result`]: crate::fmt::Result
/// [`io::Result`]: ../std/io/type.Result.html
///
/// # Examples
///
/// ```
/// use std::io::Write;
///
/// fn main() -> std::io::Result<()> {
///     let mut w = Vec::new();
///     write!(&mut w, "test")?;
///     write!(&mut w, "formatted {}", "arguments")?;
///
///     assert_eq!(w, b"testformatted arguments");
///     Ok(())
/// }
/// ```
///
/// A module can import both `std::fmt::Write` and `std::io::Write` and call `write!` on objects
/// implementing either, as objects do not typically implement both. However, the module must
/// import the traits qualified so their names do not conflict:
///
/// ```
/// use std::fmt::Write as FmtWrite;
/// use std::io::Write as IoWrite;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let mut s = String::new();
///     let mut v = Vec::new();
///
///     write!(&mut s, "{} {}", "abc", 123)?; // uses fmt::Write::write_fmt
///     write!(&mut v, "s = {:?}", s)?; // uses io::Write::write_fmt
///     assert_eq!(v, b"s = \"abc 123\"");
///     Ok(())
/// }
/// ```
///
/// Note: This macro can be used in `no_std` setups as well.
/// In a `no_std` setup you are responsible for the implementation details of the components.
///
/// ```no_run
/// # extern crate core;
/// use core::fmt::Write;
///
/// struct Example;
///
/// impl Write for Example {
///     fn write_str(&mut self, _s: &str) -> core::fmt::Result {
///          unimplemented!();
///     }
/// }
///
/// let mut m = Example{};
/// write!(&mut m, "Hello World").expect("Not written");
/// ```
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
macro_rules! write {
    ($dst:expr, $($arg:tt)*) => ($dst.write_fmt($crate::format_args!($($arg)*)))
}

/// Write formatted data into a buffer, with a newline appended.
///
/// On all platforms, the newline is the LINE FEED character (`\n`/`U+000A`) alone
/// (no additional CARRIAGE RETURN (`\r`/`U+000D`).
///
/// For more information, see [`write!`]. For information on the format string syntax, see
/// [`std::fmt`].
///
/// [`std::fmt`]: ../std/fmt/index.html
///
/// # Examples
///
/// ```
/// use std::io::{Write, Result};
///
/// fn main() -> Result<()> {
///     let mut w = Vec::new();
///     writeln!(&mut w)?;
///     writeln!(&mut w, "test")?;
///     writeln!(&mut w, "formatted {}", "arguments")?;
///
///     assert_eq!(&w[..], "\ntest\nformatted arguments\n".as_bytes());
///     Ok(())
/// }
/// ```
///
/// A module can import both `std::fmt::Write` and `std::io::Write` and call `write!` on objects
/// implementing either, as objects do not typically implement both. However, the module must
/// import the traits qualified so their names do not conflict:
///
/// ```
/// use std::fmt::Write as FmtWrite;
/// use std::io::Write as IoWrite;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let mut s = String::new();
///     let mut v = Vec::new();
///
///     writeln!(&mut s, "{} {}", "abc", 123)?; // uses fmt::Write::write_fmt
///     writeln!(&mut v, "s = {:?}", s)?; // uses io::Write::write_fmt
///     assert_eq!(v, b"s = \"abc 123\\n\"\n");
///     Ok(())
/// }
/// ```
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
#[allow_internal_unstable(format_args_nl)]
macro_rules! writeln {
    ($dst:expr $(,)?) => (
        $crate::write!($dst, "\n")
    );
    ($dst:expr, $($arg:tt)*) => (
        $dst.write_fmt($crate::format_args_nl!($($arg)*))
    );
}

/// Indicates unreachable code.
///
/// This is useful any time that the compiler can't determine that some code is unreachable. For
/// example:
///
/// * Match arms with guard conditions.
/// * Loops that dynamically terminate.
/// * Iterators that dynamically terminate.
///
/// If the determination that the code is unreachable proves incorrect, the
/// program immediately terminates with a [`panic!`].
///
/// The unsafe counterpart of this macro is the [`unreachable_unchecked`] function, which
/// will cause undefined behavior if the code is reached.
///
/// [`unreachable_unchecked`]: crate::hint::unreachable_unchecked
///
/// # Panics
///
/// This will always [`panic!`] because `unreachable!` is just a shorthand for `panic!` with a
/// fixed, specific message.
///
/// Like `panic!`, this macro has a second form for displaying custom values.
///
/// # Examples
///
/// Match arms:
///
/// ```
/// # #[allow(dead_code)]
/// fn foo(x: Option<i32>) {
///     match x {
///         Some(n) if n >= 0 => println!("Some(Non-negative)"),
///         Some(n) if n <  0 => println!("Some(Negative)"),
///         Some(_)           => unreachable!(), // compile error if commented out
///         None              => println!("None")
///     }
/// }
/// ```
///
/// Iterators:
///
/// ```
/// # #[allow(dead_code)]
/// fn divide_by_three(x: u32) -> u32 { // one of the poorest implementations of x/3
///     for i in 0.. {
///         if 3*i < i { panic!("u32 overflow"); }
///         if x < 3*i { return i-1; }
///     }
///     unreachable!("The loop should always return");
/// }
/// ```
#[cfg(not(bootstrap))]
#[macro_export]
#[rustc_builtin_macro(unreachable)]
#[allow_internal_unstable(edition_panic)]
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "unreachable_macro")]
macro_rules! unreachable {
    // Expands to either `$crate::panic::unreachable_2015` or `$crate::panic::unreachable_2021`
    // depending on the edition of the caller.
    ($($arg:tt)*) => {
        /* compiler built-in */
    };
}

/// unreachable!() macro
#[cfg(bootstrap)]
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
#[allow_internal_unstable(core_panic)]
macro_rules! unreachable {
    () => ({
        $crate::panicking::panic("internal error: entered unreachable code")
    });
    ($msg:expr $(,)?) => ({
        $crate::unreachable!("{}", $msg)
    });
    ($fmt:expr, $($arg:tt)*) => ({
        $crate::panic!($crate::concat!("internal error: entered unreachable code: ", $fmt), $($arg)*)
    });
}

/// Indicates unimplemented code by panicking with a message of "not implemented".
///
/// This allows your code to type-check, which is useful if you are prototyping or
/// implementing a trait that requires multiple methods which you don't plan to use all of.
///
/// The difference between `unimplemented!` and [`todo!`] is that while `todo!`
/// conveys an intent of implementing the functionality later and the message is "not yet
/// implemented", `unimplemented!` makes no such claims. Its message is "not implemented".
/// Also some IDEs will mark `todo!`s.
///
/// # Panics
///
/// This will always [`panic!`] because `unimplemented!` is just a shorthand for `panic!` with a
/// fixed, specific message.
///
/// Like `panic!`, this macro has a second form for displaying custom values.
///
/// # Examples
///
/// Say we have a trait `Foo`:
///
/// ```
/// trait Foo {
///     fn bar(&self) -> u8;
///     fn baz(&self);
///     fn qux(&self) -> Result<u64, ()>;
/// }
/// ```
///
/// We want to implement `Foo` for 'MyStruct', but for some reason it only makes sense
/// to implement the `bar()` function. `baz()` and `qux()` will still need to be defined
/// in our implementation of `Foo`, but we can use `unimplemented!` in their definitions
/// to allow our code to compile.
///
/// We still want to have our program stop running if the unimplemented methods are
/// reached.
///
/// ```
/// # trait Foo {
/// #     fn bar(&self) -> u8;
/// #     fn baz(&self);
/// #     fn qux(&self) -> Result<u64, ()>;
/// # }
/// struct MyStruct;
///
/// impl Foo for MyStruct {
///     fn bar(&self) -> u8 {
///         1 + 1
///     }
///
///     fn baz(&self) {
///         // It makes no sense to `baz` a `MyStruct`, so we have no logic here
///         // at all.
///         // This will display "thread 'main' panicked at 'not implemented'".
///         unimplemented!();
///     }
///
///     fn qux(&self) -> Result<u64, ()> {
///         // We have some logic here,
///         // We can add a message to unimplemented! to display our omission.
///         // This will display:
///         // "thread 'main' panicked at 'not implemented: MyStruct isn't quxable'".
///         unimplemented!("MyStruct isn't quxable");
///     }
/// }
///
/// fn main() {
///     let s = MyStruct;
///     s.bar();
/// }
/// ```
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
#[allow_internal_unstable(core_panic)]
macro_rules! unimplemented {
    () => ($crate::panicking::panic("not implemented"));
    ($($arg:tt)+) => ($crate::panic!("not implemented: {}", $crate::format_args!($($arg)+)));
}

/// Indicates unfinished code.
///
/// This can be useful if you are prototyping and are just looking to have your
/// code typecheck.
///
/// The difference between [`unimplemented!`] and `todo!` is that while `todo!` conveys
/// an intent of implementing the functionality later and the message is "not yet
/// implemented", `unimplemented!` makes no such claims. Its message is "not implemented".
/// Also some IDEs will mark `todo!`s.
///
/// # Panics
///
/// This will always [`panic!`].
///
/// # Examples
///
/// Here's an example of some in-progress code. We have a trait `Foo`:
///
/// ```
/// trait Foo {
///     fn bar(&self);
///     fn baz(&self);
/// }
/// ```
///
/// We want to implement `Foo` on one of our types, but we also want to work on
/// just `bar()` first. In order for our code to compile, we need to implement
/// `baz()`, so we can use `todo!`:
///
/// ```
/// # trait Foo {
/// #     fn bar(&self);
/// #     fn baz(&self);
/// # }
/// struct MyStruct;
///
/// impl Foo for MyStruct {
///     fn bar(&self) {
///         // implementation goes here
///     }
///
///     fn baz(&self) {
///         // let's not worry about implementing baz() for now
///         todo!();
///     }
/// }
///
/// fn main() {
///     let s = MyStruct;
///     s.bar();
///
///     // we aren't even using baz(), so this is fine.
/// }
/// ```
#[macro_export]
#[stable(feature = "todo_macro", since = "1.40.0")]
#[allow_internal_unstable(core_panic)]
macro_rules! todo {
    () => ($crate::panicking::panic("not yet implemented"));
    ($($arg:tt)+) => ($crate::panic!("not yet implemented: {}", $crate::format_args!($($arg)+)));
}

/// Definitions of built-in macros.
///
/// Most of the macro properties (stability, visibility, etc.) are taken from the source code here,
/// with exception of expansion functions transforming macro inputs into outputs,
/// those functions are provided by the compiler.
pub(crate) mod builtin {

    /// Causes compilation to fail with the given error message when encountered.
    ///
    /// This macro should be used when a crate uses a conditional compilation strategy to provide
    /// better error messages for erroneous conditions. It's the compiler-level form of [`panic!`],
    /// but emits an error during *compilation* rather than at *runtime*.
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
    #[stable(feature = "compile_error_macro", since = "1.20.0")]
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! compile_error {
        ($msg:expr $(,)?) => {{ /* compiler built-in */ }};
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
    /// [`Display`]: crate::fmt::Display
    /// [`Debug`]: crate::fmt::Debug
    /// [`fmt::Arguments`]: crate::fmt::Arguments
    /// [`std::fmt`]: ../std/fmt/index.html
    /// [`format!`]: ../std/macro.format.html
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
    #[allow_internal_unsafe]
    #[allow_internal_unstable(fmt_internals)]
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! format_args {
        ($fmt:expr) => {{ /* compiler built-in */ }};
        ($fmt:expr, $($args:tt)*) => {{ /* compiler built-in */ }};
    }

    /// Same as `format_args`, but can be used in some const contexts.
    ///
    /// This macro is used by the panic macros for the `const_panic` feature.
    ///
    /// This macro will be removed once `format_args` is allowed in const contexts.
    #[unstable(feature = "const_format_args", issue = "none")]
    #[allow_internal_unstable(fmt_internals, const_fmt_arguments_new)]
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! const_format_args {
        ($fmt:expr) => {{ /* compiler built-in */ }};
        ($fmt:expr, $($args:tt)*) => {{ /* compiler built-in */ }};
    }

    /// Same as `format_args`, but adds a newline in the end.
    #[unstable(
        feature = "format_args_nl",
        issue = "none",
        reason = "`format_args_nl` is only for internal \
                  language use and is subject to change"
    )]
    #[allow_internal_unstable(fmt_internals)]
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! format_args_nl {
        ($fmt:expr) => {{ /* compiler built-in */ }};
        ($fmt:expr, $($args:tt)*) => {{ /* compiler built-in */ }};
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
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! env {
        ($name:expr $(,)?) => {{ /* compiler built-in */ }};
        ($name:expr, $error_msg:expr $(,)?) => {{ /* compiler built-in */ }};
    }

    /// Optionally inspects an environment variable at compile time.
    ///
    /// If the named environment variable is present at compile time, this will
    /// expand into an expression of type `Option<&'static str>` whose value is
    /// `Some` of the value of the environment variable. If the environment
    /// variable is not present, then this will expand to `None`. See
    /// [`Option<T>`][Option] for more information on this type.
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
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! option_env {
        ($name:expr $(,)?) => {{ /* compiler built-in */ }};
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
    #[unstable(
        feature = "concat_idents",
        issue = "29599",
        reason = "`concat_idents` is not stable enough for use and is subject to change"
    )]
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! concat_idents {
        ($($e:ident),+ $(,)?) => {{ /* compiler built-in */ }};
    }

    /// Concatenates literals into a byte slice.
    ///
    /// This macro takes any number of comma-separated literals, and concatenates them all into
    /// one, yielding an expression of type `&[u8, _]`, which represents all of the literals
    /// concatenated left-to-right. The literals passed can be any combination of:
    ///
    /// - byte literals (`b'r'`)
    /// - byte strings (`b"Rust"`)
    /// - arrays of bytes/numbers (`[b'A', 66, b'C']`)
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(concat_bytes)]
    ///
    /// # fn main() {
    /// let s: &[u8; 6] = concat_bytes!(b'A', b"BC", [68, b'E', 70]);
    /// assert_eq!(s, b"ABCDEF");
    /// # }
    /// ```
    #[cfg(not(bootstrap))]
    #[unstable(feature = "concat_bytes", issue = "87555")]
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! concat_bytes {
        ($($e:literal),+ $(,)?) => {{ /* compiler built-in */ }};
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
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! concat {
        ($($e:expr),* $(,)?) => {{ /* compiler built-in */ }};
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
    /// # Examples
    ///
    /// ```
    /// let current_line = line!();
    /// println!("defined on line: {}", current_line);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! line {
        () => {
            /* compiler built-in */
        };
    }

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
    /// # Examples
    ///
    /// ```
    /// let current_col = column!();
    /// println!("defined on column: {}", current_col);
    /// ```
    ///
    /// `column!` counts Unicode code points, not bytes or graphemes. As a result, the first two
    /// invocations return the same value, but the third does not.
    ///
    /// ```
    /// let a = ("foobar", column!()).1;
    /// let b = ("人之初性本善", column!()).1;
    /// let c = ("f̅o̅o̅b̅a̅r̅", column!()).1; // Uses combining overline (U+0305)
    ///
    /// assert_eq!(a, b);
    /// assert_ne!(b, c);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! column {
        () => {
            /* compiler built-in */
        };
    }

    /// Expands to the file name in which it was invoked.
    ///
    /// With [`line!`] and [`column!`], these macros provide debugging information for
    /// developers about the location within the source.
    ///
    /// The expanded expression has type `&'static str`, and the returned file
    /// is not the invocation of the `file!` macro itself, but rather the
    /// first macro invocation leading up to the invocation of the `file!`
    /// macro.
    ///
    /// # Examples
    ///
    /// ```
    /// let this_file = file!();
    /// println!("defined in file: {}", this_file);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! file {
        () => {
            /* compiler built-in */
        };
    }

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
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! stringify {
        ($($t:tt)*) => {
            /* compiler built-in */
        };
    }

    /// Includes a UTF-8 encoded file as a string.
    ///
    /// The file is located relative to the current file (similarly to how
    /// modules are found). The provided path is interpreted in a platform-specific
    /// way at compile time. So, for instance, an invocation with a Windows path
    /// containing backslashes `\` would not compile correctly on Unix.
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
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! include_str {
        ($file:expr $(,)?) => {{ /* compiler built-in */ }};
    }

    /// Includes a file as a reference to a byte array.
    ///
    /// The file is located relative to the current file (similarly to how
    /// modules are found). The provided path is interpreted in a platform-specific
    /// way at compile time. So, for instance, an invocation with a Windows path
    /// containing backslashes `\` would not compile correctly on Unix.
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
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! include_bytes {
        ($file:expr $(,)?) => {{ /* compiler built-in */ }};
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
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! module_path {
        () => {
            /* compiler built-in */
        };
    }

    /// Evaluates boolean combinations of configuration flags at compile-time.
    ///
    /// In addition to the `#[cfg]` attribute, this macro is provided to allow
    /// boolean expression evaluation of configuration flags. This frequently
    /// leads to less duplicated code.
    ///
    /// The syntax given to this macro is the same syntax as the [`cfg`]
    /// attribute.
    ///
    /// `cfg!`, unlike `#[cfg]`, does not remove any code and only evaluates to true or false. For
    /// example, all blocks in an if/else expression need to be valid when `cfg!` is used for
    /// the condition, regardless of what `cfg!` is evaluating.
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
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! cfg {
        ($($cfg:tt)*) => {
            /* compiler built-in */
        };
    }

    /// Parses a file as an expression or an item according to the context.
    ///
    /// The file is located relative to the current file (similarly to how
    /// modules are found). The provided path is interpreted in a platform-specific
    /// way at compile time. So, for instance, an invocation with a Windows path
    /// containing backslashes `\` would not compile correctly on Unix.
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
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! include {
        ($file:expr $(,)?) => {{ /* compiler built-in */ }};
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
    /// Unsafe code may rely on `assert!` to enforce run-time invariants that, if
    /// violated could lead to unsafety.
    ///
    /// Other use-cases of `assert!` include testing and enforcing run-time
    /// invariants in safe code (whose violation cannot result in unsafety).
    ///
    /// # Custom Messages
    ///
    /// This macro has a second form, where a custom panic message can
    /// be provided with or without arguments for formatting. See [`std::fmt`]
    /// for syntax for this form. Expressions used as format arguments will only
    /// be evaluated if the assertion fails.
    ///
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
    #[rustc_builtin_macro]
    #[macro_export]
    #[rustc_diagnostic_item = "assert_macro"]
    #[allow_internal_unstable(core_panic, edition_panic)]
    macro_rules! assert {
        ($cond:expr $(,)?) => {{ /* compiler built-in */ }};
        ($cond:expr, $($arg:tt)+) => {{ /* compiler built-in */ }};
    }

    /// LLVM-style inline assembly.
    ///
    /// Read the [unstable book] for the usage.
    ///
    /// [unstable book]: ../unstable-book/library-features/llvm-asm.html
    #[unstable(
        feature = "llvm_asm",
        issue = "70173",
        reason = "prefer using the new asm! syntax instead"
    )]
    #[rustc_deprecated(
        since = "1.56",
        reason = "will be removed from the compiler, use asm! instead"
    )]
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! llvm_asm {
        ("assembly template"
                        : $("output"(operand),)*
                        : $("input"(operand),)*
                        : $("clobbers",)*
                        : $("options",)*) => {
            /* compiler built-in */
        };
    }

    /// Prints passed tokens into the standard output.
    #[unstable(
        feature = "log_syntax",
        issue = "29598",
        reason = "`log_syntax!` is not stable enough for use and is subject to change"
    )]
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! log_syntax {
        ($($arg:tt)*) => {
            /* compiler built-in */
        };
    }

    /// Enables or disables tracing functionality used for debugging other macros.
    #[unstable(
        feature = "trace_macros",
        issue = "29598",
        reason = "`trace_macros` is not stable enough for use and is subject to change"
    )]
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! trace_macros {
        (true) => {{ /* compiler built-in */ }};
        (false) => {{ /* compiler built-in */ }};
    }

    /// Attribute macro used to apply derive macros.
    ///
    /// See [the reference] for more info.
    ///
    /// [the reference]: ../../../reference/attributes/derive.html
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_builtin_macro]
    pub macro derive($item:item) {
        /* compiler built-in */
    }

    /// Attribute macro applied to a function to turn it into a unit test.
    ///
    /// See [the reference] for more info.
    ///
    /// [the reference]: ../../../reference/attributes/testing.html#the-test-attribute
    #[stable(feature = "rust1", since = "1.0.0")]
    #[allow_internal_unstable(test, rustc_attrs)]
    #[rustc_builtin_macro]
    pub macro test($item:item) {
        /* compiler built-in */
    }

    /// Attribute macro applied to a function to turn it into a benchmark test.
    #[unstable(
        feature = "test",
        issue = "50297",
        soft,
        reason = "`bench` is a part of custom test frameworks which are unstable"
    )]
    #[allow_internal_unstable(test, rustc_attrs)]
    #[rustc_builtin_macro]
    pub macro bench($item:item) {
        /* compiler built-in */
    }

    /// An implementation detail of the `#[test]` and `#[bench]` macros.
    #[unstable(
        feature = "custom_test_frameworks",
        issue = "50297",
        reason = "custom test frameworks are an unstable feature"
    )]
    #[allow_internal_unstable(test, rustc_attrs)]
    #[rustc_builtin_macro]
    pub macro test_case($item:item) {
        /* compiler built-in */
    }

    /// Attribute macro applied to a static to register it as a global allocator.
    ///
    /// See also [`std::alloc::GlobalAlloc`](../../../std/alloc/trait.GlobalAlloc.html).
    #[stable(feature = "global_allocator", since = "1.28.0")]
    #[allow_internal_unstable(rustc_attrs)]
    #[rustc_builtin_macro]
    pub macro global_allocator($item:item) {
        /* compiler built-in */
    }

    /// Keeps the item it's applied to if the passed path is accessible, and removes it otherwise.
    #[unstable(
        feature = "cfg_accessible",
        issue = "64797",
        reason = "`cfg_accessible` is not fully implemented"
    )]
    #[rustc_builtin_macro]
    pub macro cfg_accessible($item:item) {
        /* compiler built-in */
    }

    /// Expands all `#[cfg]` and `#[cfg_attr]` attributes in the code fragment it's applied to.
    #[unstable(
        feature = "cfg_eval",
        issue = "82679",
        reason = "`cfg_eval` is a recently implemented feature"
    )]
    #[rustc_builtin_macro]
    pub macro cfg_eval($($tt:tt)*) {
        /* compiler built-in */
    }

    /// Unstable implementation detail of the `rustc` compiler, do not use.
    #[rustc_builtin_macro]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[allow_internal_unstable(core_intrinsics, libstd_sys_internals)]
    #[rustc_deprecated(
        since = "1.52.0",
        reason = "rustc-serialize is deprecated and no longer supported"
    )]
    #[doc(hidden)] // While technically stable, using it is unstable, and deprecated. Hide it.
    pub macro RustcDecodable($item:item) {
        /* compiler built-in */
    }

    /// Unstable implementation detail of the `rustc` compiler, do not use.
    #[rustc_builtin_macro]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[allow_internal_unstable(core_intrinsics)]
    #[rustc_deprecated(
        since = "1.52.0",
        reason = "rustc-serialize is deprecated and no longer supported"
    )]
    #[doc(hidden)] // While technically stable, using it is unstable, and deprecated. Hide it.
    pub macro RustcEncodable($item:item) {
        /* compiler built-in */
    }
}
