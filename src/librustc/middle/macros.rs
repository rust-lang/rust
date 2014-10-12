// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![macro_escape]

macro_rules! memoize_expand_block(
    ($cache_map:expr, $cache_key:expr, $($param_name:ident: $param_ty:ty),*) => { {
        match ($cache_map).borrow().find(&$cache_key) {
            Some(ref result) => return (*result).clone(),
            None => {}
        }
        let result = inner($($param_name), *);
        ($cache_map).borrow_mut().insert($cache_key, result.clone());
        result
    } }
)

/// Memoizes a function using a cache that is available by evaluating the
/// `$cache_map` exression in the context of the function's arguments.
/// `$cache_key` is the expression that will be used to compute the cache key
/// for each function invocation.
///
/// The macro assumes the cache to be a RefCell containing a HashMap,
/// which is in practice how most caching in rustc is currently carried out.
///
/// # Example
///
/// ```
/// struct Context {
///     fibonacci_cache: RefCell<HashMap<uint, uint>>
/// }
///
/// memoize!(context.fibonacci_cache, n,
/// fn fibonacci(context: &Context, n: uint) -> uint {
///     match n {
///         0 | 1 => n,
///         _ => fibonacci(n - 2) + fibonacci(n - 1)
///     }
/// }
/// )
/// ```
macro_rules! memoize(
    ($cache_map:expr, $cache_key:expr,
        fn $name:ident(
            $($param_name:ident: $param_ty:ty),*
        ) -> $output_ty:ty $block:block
    ) => {
        fn $name($($param_name: $param_ty), *) -> $output_ty {
            fn inner($($param_name: $param_ty), *) -> $output_ty $block
            memoize_expand_block!($cache_map, $cache_key, $($param_name: $param_ty), *)
        }
    };

    ($cache_map:expr, $cache_key:expr,
        pub fn $name:ident(
            $($param_name:ident: $param_ty:ty),*
        ) -> $output_ty:ty $block:block
    ) => {
        pub fn $name($($param_name: $param_ty), *) -> $output_ty {
            fn inner($($param_name: $param_ty), *) -> $output_ty $block
            memoize_expand_block!($cache_map, $cache_key, $($param_name: $param_ty), *)
        }
    }
)
