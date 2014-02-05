// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#[macro_escape];

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
#[macro_export]
macro_rules! error( ($($arg:tt)*) => (log!(1u32, $($arg)*)) )
#[macro_export]
macro_rules! warn ( ($($arg:tt)*) => (log!(2u32, $($arg)*)) )
#[macro_export]
macro_rules! info ( ($($arg:tt)*) => (log!(3u32, $($arg)*)) )
#[macro_export]
macro_rules! debug( ($($arg:tt)*) => (
    if cfg!(not(ndebug)) { log!(4u32, $($arg)*) }
))

#[macro_export]
macro_rules! log_enabled(
    ($lvl:expr) => ( {
        let lvl = $lvl;
        lvl <= __log_level() && (lvl != 4 || cfg!(not(ndebug)))
    } )
)

#[macro_export]
macro_rules! fail(
    () => (
        fail!("explicit failure")
    );
    ($msg:expr) => (
        ::std::rt::begin_unwind($msg, file!(), line!())
    );
    ($fmt:expr, $($arg:tt)*) => (
        {
            // a closure can't have return type !, so we need a full
            // function to pass to format_args!, *and* we need the
            // file and line numbers right here; so an inner bare fn
            // is our only choice.
            #[inline]
            fn run_fmt(fmt: &::std::fmt::Arguments) -> ! {
                ::std::rt::begin_unwind_fmt(fmt, file!(), line!())
            }
            format_args!(run_fmt, $fmt, $($arg)*)
        }
        )
)

#[macro_export]
macro_rules! assert(
    ($cond:expr) => {
        if !$cond {
            fail!("assertion failed: {:s}", stringify!($cond))
        }
    };
    ($cond:expr, $msg:expr) => {
        if !$cond {
            fail!($msg)
        }
    };
    ($cond:expr, $( $arg:expr ),+) => {
        if !$cond {
            fail!( $($arg),+ )
        }
    }
)

#[macro_export]
macro_rules! assert_eq (
    ($given:expr , $expected:expr) => (
        {
            let given_val = &($given);
            let expected_val = &($expected);
            // check both directions of equality....
            if !((*given_val == *expected_val) &&
                 (*expected_val == *given_val)) {
                fail!("assertion failed: `(left == right) && (right == left)` \
                       (left: `{:?}`, right: `{:?}`)", *given_val, *expected_val)
            }
        }
    )
)

/// A utility macro for indicating unreachable code. It will fail if
/// executed. This is occasionally useful to put after loops that never
/// terminate normally, but instead directly return from a function.
///
/// # Example
///
/// ```rust
/// fn choose_weighted_item(v: &[Item]) -> Item {
///     assert!(!v.is_empty());
///     let mut so_far = 0u;
///     for item in v.iter() {
///         so_far += item.weight;
///         if so_far > 100 {
///             return item;
///         }
///     }
///     // The above loop always returns, so we must hint to the
///     // type checker that it isn't possible to get down here
///     unreachable!();
/// }
/// ```
#[macro_export]
macro_rules! unreachable (() => (
    fail!("internal error: entered unreachable code");
))

#[macro_export]
macro_rules! format(($($arg:tt)*) => (
    format_args!(::std::fmt::format, $($arg)*)
))
#[macro_export]
macro_rules! write(($dst:expr, $($arg:tt)*) => (
    format_args!(|args| { ::std::fmt::write($dst, args) }, $($arg)*)
))
#[macro_export]
macro_rules! writeln(($dst:expr, $($arg:tt)*) => (
    format_args!(|args| { ::std::fmt::writeln($dst, args) }, $($arg)*)
))
#[macro_export]
macro_rules! print (
    ($($arg:tt)*) => (format_args!(::std::io::stdio::print_args, $($arg)*))
)
#[macro_export]
macro_rules! println (
    ($($arg:tt)*) => (format_args!(::std::io::stdio::println_args, $($arg)*))
)

#[macro_export]
macro_rules! local_data_key (
    ($name:ident: $ty:ty) => (
        static $name: ::std::local_data::Key<$ty> = &::std::local_data::Key;
    );
    (pub $name:ident: $ty:ty) => (
        pub static $name: ::std::local_data::Key<$ty> = &::std::local_data::Key;
    )
)

#[macro_export]
macro_rules! if_ok (
    ($e:expr) => (match $e { Ok(e) => e, Err(e) => return Err(e) })
)
