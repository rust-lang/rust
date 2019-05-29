/// Panics the current thread.
///
/// For details, see `std::macros`.
#[macro_export]
#[allow_internal_unstable(core_panic, __rust_unstable_column)]
#[stable(feature = "core", since = "1.6.0")]
macro_rules! panic {
    () => (
        $crate::panic!("explicit panic")
    );
    ($msg:expr) => ({
        $crate::panicking::panic(&($msg, file!(), line!(), __rust_unstable_column!()))
    });
    ($msg:expr,) => (
        $crate::panic!($msg)
    );
    ($fmt:expr, $($arg:tt)+) => ({
        $crate::panicking::panic_fmt(format_args!($fmt, $($arg)+),
                                     &(file!(), line!(), __rust_unstable_column!()))
    });
}

/// Asserts that two expressions are equal to each other (using [`PartialEq`]).
///
/// On panic, this macro will print the values of the expressions with their
/// debug representations.
///
/// Like [`assert!`], this macro has a second form, where a custom
/// panic message can be provided.
///
/// [`PartialEq`]: cmp/trait.PartialEq.html
/// [`assert!`]: macro.assert.html
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
macro_rules! assert_eq {
    ($left:expr, $right:expr) => ({
        match (&$left, &$right) {
            (left_val, right_val) => {
                if !(*left_val == *right_val) {
                    // The reborrows below are intentional. Without them, the stack slot for the
                    // borrow is initialized even before the values are compared, leading to a
                    // noticeable slow down.
                    panic!(r#"assertion failed: `(left == right)`
  left: `{:?}`,
 right: `{:?}`"#, &*left_val, &*right_val)
                }
            }
        }
    });
    ($left:expr, $right:expr,) => ({
        $crate::assert_eq!($left, $right)
    });
    ($left:expr, $right:expr, $($arg:tt)+) => ({
        match (&($left), &($right)) {
            (left_val, right_val) => {
                if !(*left_val == *right_val) {
                    // The reborrows below are intentional. Without them, the stack slot for the
                    // borrow is initialized even before the values are compared, leading to a
                    // noticeable slow down.
                    panic!(r#"assertion failed: `(left == right)`
  left: `{:?}`,
 right: `{:?}`: {}"#, &*left_val, &*right_val,
                           format_args!($($arg)+))
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
/// [`PartialEq`]: cmp/trait.PartialEq.html
/// [`assert!`]: macro.assert.html
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
macro_rules! assert_ne {
    ($left:expr, $right:expr) => ({
        match (&$left, &$right) {
            (left_val, right_val) => {
                if *left_val == *right_val {
                    // The reborrows below are intentional. Without them, the stack slot for the
                    // borrow is initialized even before the values are compared, leading to a
                    // noticeable slow down.
                    panic!(r#"assertion failed: `(left != right)`
  left: `{:?}`,
 right: `{:?}`"#, &*left_val, &*right_val)
                }
            }
        }
    });
    ($left:expr, $right:expr,) => {
        $crate::assert_ne!($left, $right)
    };
    ($left:expr, $right:expr, $($arg:tt)+) => ({
        match (&($left), &($right)) {
            (left_val, right_val) => {
                if *left_val == *right_val {
                    // The reborrows below are intentional. Without them, the stack slot for the
                    // borrow is initialized even before the values are compared, leading to a
                    // noticeable slow down.
                    panic!(r#"assertion failed: `(left != right)`
  left: `{:?}`,
 right: `{:?}`: {}"#, &*left_val, &*right_val,
                           format_args!($($arg)+))
                }
            }
        }
    });
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
/// optimized builds by default. An optimized build will omit all
/// `debug_assert!` statements unless `-C debug-assertions` is passed to the
/// compiler. This makes `debug_assert!` useful for checks that are too
/// expensive to be present in a release build but may be helpful during
/// development.
///
/// An unchecked assertion allows a program in an inconsistent state to keep
/// running, which might have unexpected consequences but does not introduce
/// unsafety as long as this only happens in safe code. The performance cost
/// of assertions, is however, not measurable in general. Replacing [`assert!`]
/// with `debug_assert!` is thus only encouraged after thorough profiling, and
/// more importantly, only in safe code!
///
/// [`panic!`]: macro.panic.html
/// [`assert!`]: macro.assert.html
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
macro_rules! debug_assert {
    ($($arg:tt)*) => (if cfg!(debug_assertions) { assert!($($arg)*); })
}

/// Asserts that two expressions are equal to each other.
///
/// On panic, this macro will print the values of the expressions with their
/// debug representations.
///
/// Unlike [`assert_eq!`], `debug_assert_eq!` statements are only enabled in non
/// optimized builds by default. An optimized build will omit all
/// `debug_assert_eq!` statements unless `-C debug-assertions` is passed to the
/// compiler. This makes `debug_assert_eq!` useful for checks that are too
/// expensive to be present in a release build but may be helpful during
/// development.
///
/// [`assert_eq!`]: ../std/macro.assert_eq.html
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
    ($($arg:tt)*) => (if cfg!(debug_assertions) { $crate::assert_eq!($($arg)*); })
}

/// Asserts that two expressions are not equal to each other.
///
/// On panic, this macro will print the values of the expressions with their
/// debug representations.
///
/// Unlike [`assert_ne!`], `debug_assert_ne!` statements are only enabled in non
/// optimized builds by default. An optimized build will omit all
/// `debug_assert_ne!` statements unless `-C debug-assertions` is passed to the
/// compiler. This makes `debug_assert_ne!` useful for checks that are too
/// expensive to be present in a release build but may be helpful during
/// development.
///
/// [`assert_ne!`]: ../std/macro.assert_ne.html
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
    ($($arg:tt)*) => (if cfg!(debug_assertions) { $crate::assert_ne!($($arg)*); })
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
/// [`Result`]: ../std/result/enum.Result.html
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
#[doc(alias = "?")]
macro_rules! r#try {
    ($expr:expr) => (match $expr {
        $crate::result::Result::Ok(val) => val,
        $crate::result::Result::Err(err) => {
            return $crate::result::Result::Err($crate::convert::From::from(err))
        }
    });
    ($expr:expr,) => ($crate::r#try!($expr));
}

/// Writes formatted data into a buffer.
///
/// This macro accepts a format string, a list of arguments, and a 'writer'. Arguments will be
/// formatted according to the specified format string and the result will be passed to the writer.
/// The writer may be any value with a `write_fmt` method; generally this comes from an
/// implementation of either the [`std::fmt::Write`] or the [`std::io::Write`] trait. The macro
/// returns whatever the `write_fmt` method returns; commonly a [`std::fmt::Result`], or an
/// [`io::Result`].
///
/// See [`std::fmt`] for more information on the format string syntax.
///
/// [`std::fmt`]: ../std/fmt/index.html
/// [`std::fmt::Write`]: ../std/fmt/trait.Write.html
/// [`std::io::Write`]: ../std/io/trait.Write.html
/// [`std::fmt::Result`]: ../std/fmt/type.Result.html
/// [`io::Result`]: ../std/io/type.Result.html
///
/// # Examples
///
/// ```
/// use std::io::Write;
///
/// let mut w = Vec::new();
/// write!(&mut w, "test").unwrap();
/// write!(&mut w, "formatted {}", "arguments").unwrap();
///
/// assert_eq!(w, b"testformatted arguments");
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
/// let mut s = String::new();
/// let mut v = Vec::new();
/// write!(&mut s, "{} {}", "abc", 123).unwrap(); // uses fmt::Write::write_fmt
/// write!(&mut v, "s = {:?}", s).unwrap(); // uses io::Write::write_fmt
/// assert_eq!(v, b"s = \"abc 123\"");
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
    ($dst:expr, $($arg:tt)*) => ($dst.write_fmt(format_args!($($arg)*)))
}

/// Write formatted data into a buffer, with a newline appended.
///
/// On all platforms, the newline is the LINE FEED character (`\n`/`U+000A`) alone
/// (no additional CARRIAGE RETURN (`\r`/`U+000D`).
///
/// For more information, see [`write!`]. For information on the format string syntax, see
/// [`std::fmt`].
///
/// [`write!`]: macro.write.html
/// [`std::fmt`]: ../std/fmt/index.html
///
///
/// # Examples
///
/// ```
/// use std::io::Write;
///
/// let mut w = Vec::new();
/// writeln!(&mut w).unwrap();
/// writeln!(&mut w, "test").unwrap();
/// writeln!(&mut w, "formatted {}", "arguments").unwrap();
///
/// assert_eq!(&w[..], "\ntest\nformatted arguments\n".as_bytes());
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
/// let mut s = String::new();
/// let mut v = Vec::new();
/// writeln!(&mut s, "{} {}", "abc", 123).unwrap(); // uses fmt::Write::write_fmt
/// writeln!(&mut v, "s = {:?}", s).unwrap(); // uses io::Write::write_fmt
/// assert_eq!(v, b"s = \"abc 123\\n\"\n");
/// ```
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
#[allow_internal_unstable(format_args_nl)]
macro_rules! writeln {
    ($dst:expr) => (
        $crate::write!($dst, "\n")
    );
    ($dst:expr,) => (
        $crate::writeln!($dst)
    );
    ($dst:expr, $($arg:tt)*) => (
        $dst.write_fmt(format_args_nl!($($arg)*))
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
/// [`panic!`]:  ../std/macro.panic.html
/// [`unreachable_unchecked`]: ../std/hint/fn.unreachable_unchecked.html
/// [`std::hint`]: ../std/hint/index.html
///
/// # Panics
///
/// This will always [`panic!`]
///
/// [`panic!`]: ../std/macro.panic.html
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
///     unreachable!();
/// }
/// ```
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
macro_rules! unreachable {
    () => ({
        panic!("internal error: entered unreachable code")
    });
    ($msg:expr) => ({
        $crate::unreachable!("{}", $msg)
    });
    ($msg:expr,) => ({
        $crate::unreachable!($msg)
    });
    ($fmt:expr, $($arg:tt)*) => ({
        panic!(concat!("internal error: entered unreachable code: ", $fmt), $($arg)*)
    });
}

/// Indicates unfinished code.
///
/// This can be useful if you are prototyping and are just looking to have your
/// code type-check, or if you're implementing a trait that requires multiple
/// methods, and you're only planning on using one of them.
///
/// # Panics
///
/// This will always [panic!](macro.panic.html)
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
/// `baz()`, so we can use `unimplemented!`:
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
///         unimplemented!();
///     }
/// }
///
/// fn main() {
///     let s = MyStruct;
///     s.bar();
///
///     // we aren't even using baz() yet, so this is fine.
/// }
/// ```
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
macro_rules! unimplemented {
    () => (panic!("not yet implemented"));
    ($($arg:tt)+) => (panic!("not yet implemented: {}", format_args!($($arg)+)));
}

/// Indicates unfinished code.
///
/// This can be useful if you are prototyping and are just looking to have your
/// code typecheck. `todo!` works exactly like `unimplemented!`. The only
/// difference between the two macros is the name.
///
/// # Panics
///
/// This will always [panic!](macro.panic.html)
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
/// #![feature(todo_macro)]
///
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
///     // we aren't even using baz() yet, so this is fine.
/// }
/// ```
#[macro_export]
#[unstable(feature = "todo_macro", issue = "59277")]
macro_rules! todo {
    () => (panic!("not yet implemented"));
    ($($arg:tt)+) => (panic!("not yet implemented: {}", format_args!($($arg)+)));
}

/// Creates an array of [`MaybeUninit`].
///
/// This macro constructs an uninitialized array of the type `[MaybeUninit<K>; N]`.
///
/// [`MaybeUninit`]: mem/union.MaybeUninit.html
#[macro_export]
#[unstable(feature = "maybe_uninit_array", issue = "53491")]
macro_rules! uninitialized_array {
    // This `assume_init` is safe because an array of `MaybeUninit` does not
    // require initialization.
    // FIXME(#49147): Could be replaced by an array initializer, once those can
    // be any const expression.
    ($t:ty; $size:expr) => (unsafe {
        MaybeUninit::<[MaybeUninit<$t>; $size]>::uninit().assume_init()
    });
}

/// Built-in macros to the compiler itself.
///
/// These macros do not have any corresponding definition with a `macro_rules!`
/// macro, but are documented here. Their implementations can be found hardcoded
/// into libsyntax itself.
///
/// For more information, see documentation for `std`'s macros.
#[cfg(rustdoc)]
mod builtin {

    /// Causes compilation to fail with the given error message when encountered.
    ///
    /// For more information, see the documentation for [`std::compile_error!`].
    ///
    /// [`std::compile_error!`]: ../std/macro.compile_error.html
    #[stable(feature = "compile_error_macro", since = "1.20.0")]
    #[rustc_doc_only_macro]
    macro_rules! compile_error {
        ($msg:expr) => ({ /* compiler built-in */ });
        ($msg:expr,) => ({ /* compiler built-in */ });
    }

    /// Constructs parameters for the other string-formatting macros.
    ///
    /// For more information, see the documentation for [`std::format_args!`].
    ///
    /// [`std::format_args!`]: ../std/macro.format_args.html
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_doc_only_macro]
    macro_rules! format_args {
        ($fmt:expr) => ({ /* compiler built-in */ });
        ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ });
    }

    /// Inspects an environment variable at compile time.
    ///
    /// For more information, see the documentation for [`std::env!`].
    ///
    /// [`std::env!`]: ../std/macro.env.html
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_doc_only_macro]
    macro_rules! env {
        ($name:expr) => ({ /* compiler built-in */ });
        ($name:expr,) => ({ /* compiler built-in */ });
    }

    /// Optionally inspects an environment variable at compile time.
    ///
    /// For more information, see the documentation for [`std::option_env!`].
    ///
    /// [`std::option_env!`]: ../std/macro.option_env.html
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_doc_only_macro]
    macro_rules! option_env {
        ($name:expr) => ({ /* compiler built-in */ });
        ($name:expr,) => ({ /* compiler built-in */ });
    }

    /// Concatenates identifiers into one identifier.
    ///
    /// For more information, see the documentation for [`std::concat_idents!`].
    ///
    /// [`std::concat_idents!`]: ../std/macro.concat_idents.html
    #[unstable(feature = "concat_idents_macro", issue = "29599")]
    #[rustc_doc_only_macro]
    macro_rules! concat_idents {
        ($($e:ident),+) => ({ /* compiler built-in */ });
        ($($e:ident,)+) => ({ /* compiler built-in */ });
    }

    /// Concatenates literals into a static string slice.
    ///
    /// For more information, see the documentation for [`std::concat!`].
    ///
    /// [`std::concat!`]: ../std/macro.concat.html
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_doc_only_macro]
    macro_rules! concat {
        ($($e:expr),*) => ({ /* compiler built-in */ });
        ($($e:expr,)*) => ({ /* compiler built-in */ });
    }

    /// Expands to the line number on which it was invoked.
    ///
    /// For more information, see the documentation for [`std::line!`].
    ///
    /// [`std::line!`]: ../std/macro.line.html
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_doc_only_macro]
    macro_rules! line { () => ({ /* compiler built-in */ }) }

    /// Expands to the column number on which it was invoked.
    ///
    /// For more information, see the documentation for [`std::column!`].
    ///
    /// [`std::column!`]: ../std/macro.column.html
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_doc_only_macro]
    macro_rules! column { () => ({ /* compiler built-in */ }) }

    /// Expands to the file name from which it was invoked.
    ///
    /// For more information, see the documentation for [`std::file!`].
    ///
    /// [`std::file!`]: ../std/macro.file.html
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_doc_only_macro]
    macro_rules! file { () => ({ /* compiler built-in */ }) }

    /// Stringifies its arguments.
    ///
    /// For more information, see the documentation for [`std::stringify!`].
    ///
    /// [`std::stringify!`]: ../std/macro.stringify.html
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_doc_only_macro]
    macro_rules! stringify { ($($t:tt)*) => ({ /* compiler built-in */ }) }

    /// Includes a utf8-encoded file as a string.
    ///
    /// For more information, see the documentation for [`std::include_str!`].
    ///
    /// [`std::include_str!`]: ../std/macro.include_str.html
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_doc_only_macro]
    macro_rules! include_str {
        ($file:expr) => ({ /* compiler built-in */ });
        ($file:expr,) => ({ /* compiler built-in */ });
    }

    /// Includes a file as a reference to a byte array.
    ///
    /// For more information, see the documentation for [`std::include_bytes!`].
    ///
    /// [`std::include_bytes!`]: ../std/macro.include_bytes.html
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_doc_only_macro]
    macro_rules! include_bytes {
        ($file:expr) => ({ /* compiler built-in */ });
        ($file:expr,) => ({ /* compiler built-in */ });
    }

    /// Expands to a string that represents the current module path.
    ///
    /// For more information, see the documentation for [`std::module_path!`].
    ///
    /// [`std::module_path!`]: ../std/macro.module_path.html
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_doc_only_macro]
    macro_rules! module_path { () => ({ /* compiler built-in */ }) }

    /// Evaluates boolean combinations of configuration flags, at compile-time.
    ///
    /// For more information, see the documentation for [`std::cfg!`].
    ///
    /// [`std::cfg!`]: ../std/macro.cfg.html
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_doc_only_macro]
    macro_rules! cfg { ($($cfg:tt)*) => ({ /* compiler built-in */ }) }

    /// Parses a file as an expression or an item according to the context.
    ///
    /// For more information, see the documentation for [`std::include!`].
    ///
    /// [`std::include!`]: ../std/macro.include.html
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_doc_only_macro]
    macro_rules! include {
        ($file:expr) => ({ /* compiler built-in */ });
        ($file:expr,) => ({ /* compiler built-in */ });
    }

    /// Asserts that a boolean expression is `true` at runtime.
    ///
    /// For more information, see the documentation for [`std::assert!`].
    ///
    /// [`std::assert!`]: ../std/macro.assert.html
    #[rustc_doc_only_macro]
    #[stable(feature = "rust1", since = "1.0.0")]
    macro_rules! assert {
        ($cond:expr) => ({ /* compiler built-in */ });
        ($cond:expr,) => ({ /* compiler built-in */ });
        ($cond:expr, $($arg:tt)+) => ({ /* compiler built-in */ });
    }
}
