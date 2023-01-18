mod bind_instead_of_map;
mod bytecount;
mod bytes_count_to_len;
mod bytes_nth;
mod case_sensitive_file_extension_comparisons;
mod chars_cmp;
mod chars_cmp_with_unwrap;
mod chars_last_cmp;
mod chars_last_cmp_with_unwrap;
mod chars_next_cmp;
mod chars_next_cmp_with_unwrap;
mod clone_on_copy;
mod clone_on_ref_ptr;
mod cloned_instead_of_copied;
mod collapsible_str_replace;
mod err_expect;
mod expect_fun_call;
mod expect_used;
mod extend_with_drain;
mod filetype_is_file;
mod filter_map;
mod filter_map_identity;
mod filter_map_next;
mod filter_next;
mod flat_map_identity;
mod flat_map_option;
mod from_iter_instead_of_collect;
mod get_first;
mod get_last_with_len;
mod get_unwrap;
mod implicit_clone;
mod inefficient_to_string;
mod inspect_for_each;
mod into_iter_on_ref;
mod is_digit_ascii_radix;
mod iter_cloned_collect;
mod iter_count;
mod iter_kv_map;
mod iter_next_slice;
mod iter_nth;
mod iter_nth_zero;
mod iter_on_single_or_empty_collections;
mod iter_overeager_cloned;
mod iter_skip_next;
mod iter_with_drain;
mod iterator_step_by_zero;
mod manual_ok_or;
mod manual_saturating_arithmetic;
mod manual_str_repeat;
mod map_clone;
mod map_collect_result_unit;
mod map_err_ignore;
mod map_flatten;
mod map_identity;
mod map_unwrap_or;
mod mut_mutex_lock;
mod needless_collect;
mod needless_option_as_deref;
mod needless_option_take;
mod no_effect_replace;
mod obfuscated_if_else;
mod ok_expect;
mod open_options;
mod option_as_ref_deref;
mod option_map_or_none;
mod option_map_unwrap_or;
mod or_fun_call;
mod or_then_unwrap;
mod path_buf_push_overwrite;
mod range_zip_with_len;
mod repeat_once;
mod search_is_some;
mod seek_from_current;
mod seek_to_start_instead_of_rewind;
mod single_char_add_str;
mod single_char_insert_string;
mod single_char_pattern;
mod single_char_push_string;
mod skip_while_next;
mod stable_sort_primitive;
mod str_splitn;
mod string_extend_chars;
mod suspicious_map;
mod suspicious_splitn;
mod suspicious_to_owned;
mod uninit_assumed_init;
mod unit_hash;
mod unnecessary_filter_map;
mod unnecessary_fold;
mod unnecessary_iter_cloned;
mod unnecessary_join;
mod unnecessary_lazy_eval;
mod unnecessary_sort_by;
mod unnecessary_to_owned;
mod unwrap_or_else_default;
mod unwrap_used;
mod useless_asref;
mod utils;
mod vec_resize_to_zero;
mod verbose_file_reads;
mod wrong_self_convention;
mod zst_offset;

use bind_instead_of_map::BindInsteadOfMap;
use clippy_utils::consts::{constant, Constant};
use clippy_utils::diagnostics::{span_lint, span_lint_and_help};
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::ty::{contains_ty_adt_constructor_opaque, implements_trait, is_copy, is_type_diagnostic_item};
use clippy_utils::{contains_return, is_bool, is_trait_method, iter_input_pats, return_ty};
use if_chain::if_chain;
use rustc_hir as hir;
use rustc_hir::{Expr, ExprKind, TraitItem, TraitItemKind};
use rustc_hir_analysis::hir_ty_to_ty;
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::{self, TraitRef, Ty};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{sym, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usages of `cloned()` on an `Iterator` or `Option` where
    /// `copied()` could be used instead.
    ///
    /// ### Why is this bad?
    /// `copied()` is better because it guarantees that the type being cloned
    /// implements `Copy`.
    ///
    /// ### Example
    /// ```rust
    /// [1, 2, 3].iter().cloned();
    /// ```
    /// Use instead:
    /// ```rust
    /// [1, 2, 3].iter().copied();
    /// ```
    #[clippy::version = "1.53.0"]
    pub CLONED_INSTEAD_OF_COPIED,
    pedantic,
    "used `cloned` where `copied` could be used instead"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for consecutive calls to `str::replace` (2 or more)
    /// that can be collapsed into a single call.
    ///
    /// ### Why is this bad?
    /// Consecutive `str::replace` calls scan the string multiple times
    /// with repetitive code.
    ///
    /// ### Example
    /// ```rust
    /// let hello = "hesuo worpd"
    ///     .replace('s', "l")
    ///     .replace("u", "l")
    ///     .replace('p', "l");
    /// ```
    /// Use instead:
    /// ```rust
    /// let hello = "hesuo worpd".replace(['s', 'u', 'p'], "l");
    /// ```
    #[clippy::version = "1.65.0"]
    pub COLLAPSIBLE_STR_REPLACE,
    perf,
    "collapse consecutive calls to str::replace (2 or more) into a single call"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `_.cloned().<func>()` where call to `.cloned()` can be postponed.
    ///
    /// ### Why is this bad?
    /// It's often inefficient to clone all elements of an iterator, when eventually, only some
    /// of them will be consumed.
    ///
    /// ### Known Problems
    /// This `lint` removes the side of effect of cloning items in the iterator.
    /// A code that relies on that side-effect could fail.
    ///
    /// ### Examples
    /// ```rust
    /// # let vec = vec!["string".to_string()];
    /// vec.iter().cloned().take(10);
    /// vec.iter().cloned().last();
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # let vec = vec!["string".to_string()];
    /// vec.iter().take(10).cloned();
    /// vec.iter().last().cloned();
    /// ```
    #[clippy::version = "1.60.0"]
    pub ITER_OVEREAGER_CLONED,
    perf,
    "using `cloned()` early with `Iterator::iter()` can lead to some performance inefficiencies"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usages of `Iterator::flat_map()` where `filter_map()` could be
    /// used instead.
    ///
    /// ### Why is this bad?
    /// When applicable, `filter_map()` is more clear since it shows that
    /// `Option` is used to produce 0 or 1 items.
    ///
    /// ### Example
    /// ```rust
    /// let nums: Vec<i32> = ["1", "2", "whee!"].iter().flat_map(|x| x.parse().ok()).collect();
    /// ```
    /// Use instead:
    /// ```rust
    /// let nums: Vec<i32> = ["1", "2", "whee!"].iter().filter_map(|x| x.parse().ok()).collect();
    /// ```
    #[clippy::version = "1.53.0"]
    pub FLAT_MAP_OPTION,
    pedantic,
    "used `flat_map` where `filter_map` could be used instead"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `.unwrap()` or `.unwrap_err()` calls on `Result`s and `.unwrap()` call on `Option`s.
    ///
    /// ### Why is this bad?
    /// It is better to handle the `None` or `Err` case,
    /// or at least call `.expect(_)` with a more helpful message. Still, for a lot of
    /// quick-and-dirty code, `unwrap` is a good choice, which is why this lint is
    /// `Allow` by default.
    ///
    /// `result.unwrap()` will let the thread panic on `Err` values.
    /// Normally, you want to implement more sophisticated error handling,
    /// and propagate errors upwards with `?` operator.
    ///
    /// Even if you want to panic on errors, not all `Error`s implement good
    /// messages on display. Therefore, it may be beneficial to look at the places
    /// where they may get displayed. Activate this lint to do just that.
    ///
    /// ### Examples
    /// ```rust
    /// # let option = Some(1);
    /// # let result: Result<usize, ()> = Ok(1);
    /// option.unwrap();
    /// result.unwrap();
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # let option = Some(1);
    /// # let result: Result<usize, ()> = Ok(1);
    /// option.expect("more helpful message");
    /// result.expect("more helpful message");
    /// ```
    ///
    /// If [expect_used](#expect_used) is enabled, instead:
    /// ```rust,ignore
    /// # let option = Some(1);
    /// # let result: Result<usize, ()> = Ok(1);
    /// option?;
    ///
    /// // or
    ///
    /// result?;
    /// ```
    #[clippy::version = "1.45.0"]
    pub UNWRAP_USED,
    restriction,
    "using `.unwrap()` on `Result` or `Option`, which should at least get a better message using `expect()`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `.expect()` or `.expect_err()` calls on `Result`s and `.expect()` call on `Option`s.
    ///
    /// ### Why is this bad?
    /// Usually it is better to handle the `None` or `Err` case.
    /// Still, for a lot of quick-and-dirty code, `expect` is a good choice, which is why
    /// this lint is `Allow` by default.
    ///
    /// `result.expect()` will let the thread panic on `Err`
    /// values. Normally, you want to implement more sophisticated error handling,
    /// and propagate errors upwards with `?` operator.
    ///
    /// ### Examples
    /// ```rust,ignore
    /// # let option = Some(1);
    /// # let result: Result<usize, ()> = Ok(1);
    /// option.expect("one");
    /// result.expect("one");
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// # let option = Some(1);
    /// # let result: Result<usize, ()> = Ok(1);
    /// option?;
    ///
    /// // or
    ///
    /// result?;
    /// ```
    #[clippy::version = "1.45.0"]
    pub EXPECT_USED,
    restriction,
    "using `.expect()` on `Result` or `Option`, which might be better handled"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for methods that should live in a trait
    /// implementation of a `std` trait (see [llogiq's blog
    /// post](http://llogiq.github.io/2015/07/30/traits.html) for further
    /// information) instead of an inherent implementation.
    ///
    /// ### Why is this bad?
    /// Implementing the traits improve ergonomics for users of
    /// the code, often with very little cost. Also people seeing a `mul(...)`
    /// method
    /// may expect `*` to work equally, so you should have good reason to disappoint
    /// them.
    ///
    /// ### Example
    /// ```rust
    /// struct X;
    /// impl X {
    ///     fn add(&self, other: &X) -> X {
    ///         // ..
    /// # X
    ///     }
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub SHOULD_IMPLEMENT_TRAIT,
    style,
    "defining a method that should be implementing a std trait"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for methods with certain name prefixes and which
    /// doesn't match how self is taken. The actual rules are:
    ///
    /// |Prefix |Postfix     |`self` taken                   | `self` type  |
    /// |-------|------------|-------------------------------|--------------|
    /// |`as_`  | none       |`&self` or `&mut self`         | any          |
    /// |`from_`| none       | none                          | any          |
    /// |`into_`| none       |`self`                         | any          |
    /// |`is_`  | none       |`&mut self` or `&self` or none | any          |
    /// |`to_`  | `_mut`     |`&mut self`                    | any          |
    /// |`to_`  | not `_mut` |`self`                         | `Copy`       |
    /// |`to_`  | not `_mut` |`&self`                        | not `Copy`   |
    ///
    /// Note: Clippy doesn't trigger methods with `to_` prefix in:
    /// - Traits definition.
    /// Clippy can not tell if a type that implements a trait is `Copy` or not.
    /// - Traits implementation, when `&self` is taken.
    /// The method signature is controlled by the trait and often `&self` is required for all types that implement the trait
    /// (see e.g. the `std::string::ToString` trait).
    ///
    /// Clippy allows `Pin<&Self>` and `Pin<&mut Self>` if `&self` and `&mut self` is required.
    ///
    /// Please find more info here:
    /// https://rust-lang.github.io/api-guidelines/naming.html#ad-hoc-conversions-follow-as_-to_-into_-conventions-c-conv
    ///
    /// ### Why is this bad?
    /// Consistency breeds readability. If you follow the
    /// conventions, your users won't be surprised that they, e.g., need to supply a
    /// mutable reference to a `as_..` function.
    ///
    /// ### Example
    /// ```rust
    /// # struct X;
    /// impl X {
    ///     fn as_str(self) -> &'static str {
    ///         // ..
    /// # ""
    ///     }
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub WRONG_SELF_CONVENTION,
    style,
    "defining a method named with an established prefix (like \"into_\") that takes `self` with the wrong convention"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `ok().expect(..)`.
    ///
    /// ### Why is this bad?
    /// Because you usually call `expect()` on the `Result`
    /// directly to get a better error message.
    ///
    /// ### Known problems
    /// The error type needs to implement `Debug`
    ///
    /// ### Example
    /// ```rust
    /// # let x = Ok::<_, ()>(());
    /// x.ok().expect("why did I do this again?");
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # let x = Ok::<_, ()>(());
    /// x.expect("why did I do this again?");
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub OK_EXPECT,
    style,
    "using `ok().expect()`, which gives worse error messages than calling `expect` directly on the Result"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `.err().expect()` calls on the `Result` type.
    ///
    /// ### Why is this bad?
    /// `.expect_err()` can be called directly to avoid the extra type conversion from `err()`.
    ///
    /// ### Example
    /// ```should_panic
    /// let x: Result<u32, &str> = Ok(10);
    /// x.err().expect("Testing err().expect()");
    /// ```
    /// Use instead:
    /// ```should_panic
    /// let x: Result<u32, &str> = Ok(10);
    /// x.expect_err("Testing expect_err");
    /// ```
    #[clippy::version = "1.62.0"]
    pub ERR_EXPECT,
    style,
    r#"using `.err().expect("")` when `.expect_err("")` can be used"#
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usages of `_.unwrap_or_else(Default::default)` on `Option` and
    /// `Result` values.
    ///
    /// ### Why is this bad?
    /// Readability, these can be written as `_.unwrap_or_default`, which is
    /// simpler and more concise.
    ///
    /// ### Examples
    /// ```rust
    /// # let x = Some(1);
    /// x.unwrap_or_else(Default::default);
    /// x.unwrap_or_else(u32::default);
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # let x = Some(1);
    /// x.unwrap_or_default();
    /// ```
    #[clippy::version = "1.56.0"]
    pub UNWRAP_OR_ELSE_DEFAULT,
    style,
    "using `.unwrap_or_else(Default::default)`, which is more succinctly expressed as `.unwrap_or_default()`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `option.map(_).unwrap_or(_)` or `option.map(_).unwrap_or_else(_)` or
    /// `result.map(_).unwrap_or_else(_)`.
    ///
    /// ### Why is this bad?
    /// Readability, these can be written more concisely (resp.) as
    /// `option.map_or(_, _)`, `option.map_or_else(_, _)` and `result.map_or_else(_, _)`.
    ///
    /// ### Known problems
    /// The order of the arguments is not in execution order
    ///
    /// ### Examples
    /// ```rust
    /// # let option = Some(1);
    /// # let result: Result<usize, ()> = Ok(1);
    /// # fn some_function(foo: ()) -> usize { 1 }
    /// option.map(|a| a + 1).unwrap_or(0);
    /// result.map(|a| a + 1).unwrap_or_else(some_function);
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # let option = Some(1);
    /// # let result: Result<usize, ()> = Ok(1);
    /// # fn some_function(foo: ()) -> usize { 1 }
    /// option.map_or(0, |a| a + 1);
    /// result.map_or_else(some_function, |a| a + 1);
    /// ```
    #[clippy::version = "1.45.0"]
    pub MAP_UNWRAP_OR,
    pedantic,
    "using `.map(f).unwrap_or(a)` or `.map(f).unwrap_or_else(func)`, which are more succinctly expressed as `map_or(a, f)` or `map_or_else(a, f)`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `_.map_or(None, _)`.
    ///
    /// ### Why is this bad?
    /// Readability, this can be written more concisely as
    /// `_.and_then(_)`.
    ///
    /// ### Known problems
    /// The order of the arguments is not in execution order.
    ///
    /// ### Example
    /// ```rust
    /// # let opt = Some(1);
    /// opt.map_or(None, |a| Some(a + 1));
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # let opt = Some(1);
    /// opt.and_then(|a| Some(a + 1));
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub OPTION_MAP_OR_NONE,
    style,
    "using `Option.map_or(None, f)`, which is more succinctly expressed as `and_then(f)`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `_.map_or(None, Some)`.
    ///
    /// ### Why is this bad?
    /// Readability, this can be written more concisely as
    /// `_.ok()`.
    ///
    /// ### Example
    /// ```rust
    /// # let r: Result<u32, &str> = Ok(1);
    /// assert_eq!(Some(1), r.map_or(None, Some));
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # let r: Result<u32, &str> = Ok(1);
    /// assert_eq!(Some(1), r.ok());
    /// ```
    #[clippy::version = "1.44.0"]
    pub RESULT_MAP_OR_INTO_OPTION,
    style,
    "using `Result.map_or(None, Some)`, which is more succinctly expressed as `ok()`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `_.and_then(|x| Some(y))`, `_.and_then(|x| Ok(y))` or
    /// `_.or_else(|x| Err(y))`.
    ///
    /// ### Why is this bad?
    /// Readability, this can be written more concisely as
    /// `_.map(|x| y)` or `_.map_err(|x| y)`.
    ///
    /// ### Example
    /// ```rust
    /// # fn opt() -> Option<&'static str> { Some("42") }
    /// # fn res() -> Result<&'static str, &'static str> { Ok("42") }
    /// let _ = opt().and_then(|s| Some(s.len()));
    /// let _ = res().and_then(|s| if s.len() == 42 { Ok(10) } else { Ok(20) });
    /// let _ = res().or_else(|s| if s.len() == 42 { Err(10) } else { Err(20) });
    /// ```
    ///
    /// The correct use would be:
    ///
    /// ```rust
    /// # fn opt() -> Option<&'static str> { Some("42") }
    /// # fn res() -> Result<&'static str, &'static str> { Ok("42") }
    /// let _ = opt().map(|s| s.len());
    /// let _ = res().map(|s| if s.len() == 42 { 10 } else { 20 });
    /// let _ = res().map_err(|s| if s.len() == 42 { 10 } else { 20 });
    /// ```
    #[clippy::version = "1.45.0"]
    pub BIND_INSTEAD_OF_MAP,
    complexity,
    "using `Option.and_then(|x| Some(y))`, which is more succinctly expressed as `map(|x| y)`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `_.filter(_).next()`.
    ///
    /// ### Why is this bad?
    /// Readability, this can be written more concisely as
    /// `_.find(_)`.
    ///
    /// ### Example
    /// ```rust
    /// # let vec = vec![1];
    /// vec.iter().filter(|x| **x == 0).next();
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # let vec = vec![1];
    /// vec.iter().find(|x| **x == 0);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub FILTER_NEXT,
    complexity,
    "using `filter(p).next()`, which is more succinctly expressed as `.find(p)`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `_.skip_while(condition).next()`.
    ///
    /// ### Why is this bad?
    /// Readability, this can be written more concisely as
    /// `_.find(!condition)`.
    ///
    /// ### Example
    /// ```rust
    /// # let vec = vec![1];
    /// vec.iter().skip_while(|x| **x == 0).next();
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # let vec = vec![1];
    /// vec.iter().find(|x| **x != 0);
    /// ```
    #[clippy::version = "1.42.0"]
    pub SKIP_WHILE_NEXT,
    complexity,
    "using `skip_while(p).next()`, which is more succinctly expressed as `.find(!p)`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `_.map(_).flatten(_)` on `Iterator` and `Option`
    ///
    /// ### Why is this bad?
    /// Readability, this can be written more concisely as
    /// `_.flat_map(_)` for `Iterator` or `_.and_then(_)` for `Option`
    ///
    /// ### Example
    /// ```rust
    /// let vec = vec![vec![1]];
    /// let opt = Some(5);
    ///
    /// vec.iter().map(|x| x.iter()).flatten();
    /// opt.map(|x| Some(x * 2)).flatten();
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # let vec = vec![vec![1]];
    /// # let opt = Some(5);
    /// vec.iter().flat_map(|x| x.iter());
    /// opt.and_then(|x| Some(x * 2));
    /// ```
    #[clippy::version = "1.31.0"]
    pub MAP_FLATTEN,
    complexity,
    "using combinations of `flatten` and `map` which can usually be written as a single method call"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `_.filter(_).map(_)` that can be written more simply
    /// as `filter_map(_)`.
    ///
    /// ### Why is this bad?
    /// Redundant code in the `filter` and `map` operations is poor style and
    /// less performant.
    ///
     /// ### Example
    /// ```rust
    /// # #![allow(unused)]
    /// (0_i32..10)
    ///     .filter(|n| n.checked_add(1).is_some())
    ///     .map(|n| n.checked_add(1).unwrap());
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # #[allow(unused)]
    /// (0_i32..10).filter_map(|n| n.checked_add(1));
    /// ```
    #[clippy::version = "1.51.0"]
    pub MANUAL_FILTER_MAP,
    complexity,
    "using `_.filter(_).map(_)` in a way that can be written more simply as `filter_map(_)`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `_.find(_).map(_)` that can be written more simply
    /// as `find_map(_)`.
    ///
    /// ### Why is this bad?
    /// Redundant code in the `find` and `map` operations is poor style and
    /// less performant.
    ///
     /// ### Example
    /// ```rust
    /// (0_i32..10)
    ///     .find(|n| n.checked_add(1).is_some())
    ///     .map(|n| n.checked_add(1).unwrap());
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// (0_i32..10).find_map(|n| n.checked_add(1));
    /// ```
    #[clippy::version = "1.51.0"]
    pub MANUAL_FIND_MAP,
    complexity,
    "using `_.find(_).map(_)` in a way that can be written more simply as `find_map(_)`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `_.filter_map(_).next()`.
    ///
    /// ### Why is this bad?
    /// Readability, this can be written more concisely as
    /// `_.find_map(_)`.
    ///
    /// ### Example
    /// ```rust
    ///  (0..3).filter_map(|x| if x == 2 { Some(x) } else { None }).next();
    /// ```
    /// Can be written as
    ///
    /// ```rust
    ///  (0..3).find_map(|x| if x == 2 { Some(x) } else { None });
    /// ```
    #[clippy::version = "1.36.0"]
    pub FILTER_MAP_NEXT,
    pedantic,
    "using combination of `filter_map` and `next` which can usually be written as a single method call"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `flat_map(|x| x)`.
    ///
    /// ### Why is this bad?
    /// Readability, this can be written more concisely by using `flatten`.
    ///
    /// ### Example
    /// ```rust
    /// # let iter = vec![vec![0]].into_iter();
    /// iter.flat_map(|x| x);
    /// ```
    /// Can be written as
    /// ```rust
    /// # let iter = vec![vec![0]].into_iter();
    /// iter.flatten();
    /// ```
    #[clippy::version = "1.39.0"]
    pub FLAT_MAP_IDENTITY,
    complexity,
    "call to `flat_map` where `flatten` is sufficient"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for an iterator or string search (such as `find()`,
    /// `position()`, or `rposition()`) followed by a call to `is_some()` or `is_none()`.
    ///
    /// ### Why is this bad?
    /// Readability, this can be written more concisely as:
    /// * `_.any(_)`, or `_.contains(_)` for `is_some()`,
    /// * `!_.any(_)`, or `!_.contains(_)` for `is_none()`.
    ///
    /// ### Example
    /// ```rust
    /// # #![allow(unused)]
    /// let vec = vec![1];
    /// vec.iter().find(|x| **x == 0).is_some();
    ///
    /// "hello world".find("world").is_none();
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let vec = vec![1];
    /// vec.iter().any(|x| *x == 0);
    ///
    /// # #[allow(unused)]
    /// !"hello world".contains("world");
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub SEARCH_IS_SOME,
    complexity,
    "using an iterator or string search followed by `is_some()` or `is_none()`, which is more succinctly expressed as a call to `any()` or `contains()` (with negation in case of `is_none()`)"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `.chars().next()` on a `str` to check
    /// if it starts with a given char.
    ///
    /// ### Why is this bad?
    /// Readability, this can be written more concisely as
    /// `_.starts_with(_)`.
    ///
    /// ### Example
    /// ```rust
    /// let name = "foo";
    /// if name.chars().next() == Some('_') {};
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let name = "foo";
    /// if name.starts_with('_') {};
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub CHARS_NEXT_CMP,
    style,
    "using `.chars().next()` to check if a string starts with a char"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to `.or(foo(..))`, `.unwrap_or(foo(..))`,
    /// `.or_insert(foo(..))` etc., and suggests to use `.or_else(|| foo(..))`,
    /// `.unwrap_or_else(|| foo(..))`, `.unwrap_or_default()` or `.or_default()`
    /// etc. instead.
    ///
    /// ### Why is this bad?
    /// The function will always be called. This is only bad if it allocates or
    /// does some non-trivial amount of work.
    ///
    /// ### Known problems
    /// If the function has side-effects, not calling it will change the
    /// semantic of the program, but you shouldn't rely on that.
    ///
    /// The lint also cannot figure out whether the function you call is
    /// actually expensive to call or not.
    ///
    /// ### Example
    /// ```rust
    /// # let foo = Some(String::new());
    /// foo.unwrap_or(String::from("empty"));
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # let foo = Some(String::new());
    /// foo.unwrap_or_else(|| String::from("empty"));
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub OR_FUN_CALL,
    nursery,
    "using any `*or` method with a function call, which suggests `*or_else`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `.or(…).unwrap()` calls to Options and Results.
    ///
    /// ### Why is this bad?
    /// You should use `.unwrap_or(…)` instead for clarity.
    ///
    /// ### Example
    /// ```rust
    /// # let fallback = "fallback";
    /// // Result
    /// # type Error = &'static str;
    /// # let result: Result<&str, Error> = Err("error");
    /// let value = result.or::<Error>(Ok(fallback)).unwrap();
    ///
    /// // Option
    /// # let option: Option<&str> = None;
    /// let value = option.or(Some(fallback)).unwrap();
    /// ```
    /// Use instead:
    /// ```rust
    /// # let fallback = "fallback";
    /// // Result
    /// # let result: Result<&str, &str> = Err("error");
    /// let value = result.unwrap_or(fallback);
    ///
    /// // Option
    /// # let option: Option<&str> = None;
    /// let value = option.unwrap_or(fallback);
    /// ```
    #[clippy::version = "1.61.0"]
    pub OR_THEN_UNWRAP,
    complexity,
    "checks for `.or(…).unwrap()` calls to Options and Results."
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to `.expect(&format!(...))`, `.expect(foo(..))`,
    /// etc., and suggests to use `unwrap_or_else` instead
    ///
    /// ### Why is this bad?
    /// The function will always be called.
    ///
    /// ### Known problems
    /// If the function has side-effects, not calling it will
    /// change the semantics of the program, but you shouldn't rely on that anyway.
    ///
    /// ### Example
    /// ```rust
    /// # let foo = Some(String::new());
    /// # let err_code = "418";
    /// # let err_msg = "I'm a teapot";
    /// foo.expect(&format!("Err {}: {}", err_code, err_msg));
    ///
    /// // or
    ///
    /// # let foo = Some(String::new());
    /// foo.expect(format!("Err {}: {}", err_code, err_msg).as_str());
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # let foo = Some(String::new());
    /// # let err_code = "418";
    /// # let err_msg = "I'm a teapot";
    /// foo.unwrap_or_else(|| panic!("Err {}: {}", err_code, err_msg));
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub EXPECT_FUN_CALL,
    perf,
    "using any `expect` method with a function call"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `.clone()` on a `Copy` type.
    ///
    /// ### Why is this bad?
    /// The only reason `Copy` types implement `Clone` is for
    /// generics, not for using the `clone` method on a concrete type.
    ///
    /// ### Example
    /// ```rust
    /// 42u64.clone();
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub CLONE_ON_COPY,
    complexity,
    "using `clone` on a `Copy` type"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `.clone()` on a ref-counted pointer,
    /// (`Rc`, `Arc`, `rc::Weak`, or `sync::Weak`), and suggests calling Clone via unified
    /// function syntax instead (e.g., `Rc::clone(foo)`).
    ///
    /// ### Why is this bad?
    /// Calling '.clone()' on an Rc, Arc, or Weak
    /// can obscure the fact that only the pointer is being cloned, not the underlying
    /// data.
    ///
    /// ### Example
    /// ```rust
    /// # use std::rc::Rc;
    /// let x = Rc::new(1);
    ///
    /// x.clone();
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # use std::rc::Rc;
    /// # let x = Rc::new(1);
    /// Rc::clone(&x);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub CLONE_ON_REF_PTR,
    restriction,
    "using 'clone' on a ref-counted pointer"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `.clone()` on an `&&T`.
    ///
    /// ### Why is this bad?
    /// Cloning an `&&T` copies the inner `&T`, instead of
    /// cloning the underlying `T`.
    ///
    /// ### Example
    /// ```rust
    /// fn main() {
    ///     let x = vec![1];
    ///     let y = &&x;
    ///     let z = y.clone();
    ///     println!("{:p} {:p}", *y, z); // prints out the same pointer
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub CLONE_DOUBLE_REF,
    correctness,
    "using `clone` on `&&T`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `.to_string()` on an `&&T` where
    /// `T` implements `ToString` directly (like `&&str` or `&&String`).
    ///
    /// ### Why is this bad?
    /// This bypasses the specialized implementation of
    /// `ToString` and instead goes through the more expensive string formatting
    /// facilities.
    ///
    /// ### Example
    /// ```rust
    /// // Generic implementation for `T: Display` is used (slow)
    /// ["foo", "bar"].iter().map(|s| s.to_string());
    ///
    /// // OK, the specialized impl is used
    /// ["foo", "bar"].iter().map(|&s| s.to_string());
    /// ```
    #[clippy::version = "1.40.0"]
    pub INEFFICIENT_TO_STRING,
    pedantic,
    "using `to_string` on `&&T` where `T: ToString`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `new` not returning a type that contains `Self`.
    ///
    /// ### Why is this bad?
    /// As a convention, `new` methods are used to make a new
    /// instance of a type.
    ///
    /// ### Example
    /// In an impl block:
    /// ```rust
    /// # struct Foo;
    /// # struct NotAFoo;
    /// impl Foo {
    ///     fn new() -> NotAFoo {
    /// # NotAFoo
    ///     }
    /// }
    /// ```
    ///
    /// ```rust
    /// # struct Foo;
    /// struct Bar(Foo);
    /// impl Foo {
    ///     // Bad. The type name must contain `Self`
    ///     fn new() -> Bar {
    /// # Bar(Foo)
    ///     }
    /// }
    /// ```
    ///
    /// ```rust
    /// # struct Foo;
    /// # struct FooError;
    /// impl Foo {
    ///     // Good. Return type contains `Self`
    ///     fn new() -> Result<Foo, FooError> {
    /// # Ok(Foo)
    ///     }
    /// }
    /// ```
    ///
    /// Or in a trait definition:
    /// ```rust
    /// pub trait Trait {
    ///     // Bad. The type name must contain `Self`
    ///     fn new();
    /// }
    /// ```
    ///
    /// ```rust
    /// pub trait Trait {
    ///     // Good. Return type contains `Self`
    ///     fn new() -> Self;
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub NEW_RET_NO_SELF,
    style,
    "not returning type containing `Self` in a `new` method"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for string methods that receive a single-character
    /// `str` as an argument, e.g., `_.split("x")`.
    ///
    /// ### Why is this bad?
    /// Performing these methods using a `char` is faster than
    /// using a `str`.
    ///
    /// ### Known problems
    /// Does not catch multi-byte unicode characters.
    ///
    /// ### Example
    /// ```rust,ignore
    /// _.split("x");
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// _.split('x');
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub SINGLE_CHAR_PATTERN,
    perf,
    "using a single-character str where a char could be used, e.g., `_.split(\"x\")`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calling `.step_by(0)` on iterators which panics.
    ///
    /// ### Why is this bad?
    /// This very much looks like an oversight. Use `panic!()` instead if you
    /// actually intend to panic.
    ///
    /// ### Example
    /// ```rust,should_panic
    /// for x in (0..100).step_by(0) {
    ///     //..
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub ITERATOR_STEP_BY_ZERO,
    correctness,
    "using `Iterator::step_by(0)`, which will panic at runtime"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for indirect collection of populated `Option`
    ///
    /// ### Why is this bad?
    /// `Option` is like a collection of 0-1 things, so `flatten`
    /// automatically does this without suspicious-looking `unwrap` calls.
    ///
    /// ### Example
    /// ```rust
    /// let _ = std::iter::empty::<Option<i32>>().filter(Option::is_some).map(Option::unwrap);
    /// ```
    /// Use instead:
    /// ```rust
    /// let _ = std::iter::empty::<Option<i32>>().flatten();
    /// ```
    #[clippy::version = "1.53.0"]
    pub OPTION_FILTER_MAP,
    complexity,
    "filtering `Option` for `Some` then force-unwrapping, which can be one type-safe operation"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the use of `iter.nth(0)`.
    ///
    /// ### Why is this bad?
    /// `iter.next()` is equivalent to
    /// `iter.nth(0)`, as they both consume the next element,
    ///  but is more readable.
    ///
    /// ### Example
    /// ```rust
    /// # use std::collections::HashSet;
    /// # let mut s = HashSet::new();
    /// # s.insert(1);
    /// let x = s.iter().nth(0);
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # use std::collections::HashSet;
    /// # let mut s = HashSet::new();
    /// # s.insert(1);
    /// let x = s.iter().next();
    /// ```
    #[clippy::version = "1.42.0"]
    pub ITER_NTH_ZERO,
    style,
    "replace `iter.nth(0)` with `iter.next()`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for use of `.iter().nth()` (and the related
    /// `.iter_mut().nth()`) on standard library types with *O*(1) element access.
    ///
    /// ### Why is this bad?
    /// `.get()` and `.get_mut()` are more efficient and more
    /// readable.
    ///
    /// ### Example
    /// ```rust
    /// let some_vec = vec![0, 1, 2, 3];
    /// let bad_vec = some_vec.iter().nth(3);
    /// let bad_slice = &some_vec[..].iter().nth(3);
    /// ```
    /// The correct use would be:
    /// ```rust
    /// let some_vec = vec![0, 1, 2, 3];
    /// let bad_vec = some_vec.get(3);
    /// let bad_slice = &some_vec[..].get(3);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub ITER_NTH,
    perf,
    "using `.iter().nth()` on a standard library type with O(1) element access"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for use of `.skip(x).next()` on iterators.
    ///
    /// ### Why is this bad?
    /// `.nth(x)` is cleaner
    ///
    /// ### Example
    /// ```rust
    /// let some_vec = vec![0, 1, 2, 3];
    /// let bad_vec = some_vec.iter().skip(3).next();
    /// let bad_slice = &some_vec[..].iter().skip(3).next();
    /// ```
    /// The correct use would be:
    /// ```rust
    /// let some_vec = vec![0, 1, 2, 3];
    /// let bad_vec = some_vec.iter().nth(3);
    /// let bad_slice = &some_vec[..].iter().nth(3);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub ITER_SKIP_NEXT,
    style,
    "using `.skip(x).next()` on an iterator"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for use of `.drain(..)` on `Vec` and `VecDeque` for iteration.
    ///
    /// ### Why is this bad?
    /// `.into_iter()` is simpler with better performance.
    ///
    /// ### Example
    /// ```rust
    /// # use std::collections::HashSet;
    /// let mut foo = vec![0, 1, 2, 3];
    /// let bar: HashSet<usize> = foo.drain(..).collect();
    /// ```
    /// Use instead:
    /// ```rust
    /// # use std::collections::HashSet;
    /// let foo = vec![0, 1, 2, 3];
    /// let bar: HashSet<usize> = foo.into_iter().collect();
    /// ```
    #[clippy::version = "1.61.0"]
    pub ITER_WITH_DRAIN,
    nursery,
    "replace `.drain(..)` with `.into_iter()`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for using `x.get(x.len() - 1)` instead of
    /// `x.last()`.
    ///
    /// ### Why is this bad?
    /// Using `x.last()` is easier to read and has the same
    /// result.
    ///
    /// Note that using `x[x.len() - 1]` is semantically different from
    /// `x.last()`.  Indexing into the array will panic on out-of-bounds
    /// accesses, while `x.get()` and `x.last()` will return `None`.
    ///
    /// There is another lint (get_unwrap) that covers the case of using
    /// `x.get(index).unwrap()` instead of `x[index]`.
    ///
    /// ### Example
    /// ```rust
    /// let x = vec![2, 3, 5];
    /// let last_element = x.get(x.len() - 1);
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let x = vec![2, 3, 5];
    /// let last_element = x.last();
    /// ```
    #[clippy::version = "1.37.0"]
    pub GET_LAST_WITH_LEN,
    complexity,
    "Using `x.get(x.len() - 1)` when `x.last()` is correct and simpler"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for use of `.get().unwrap()` (or
    /// `.get_mut().unwrap`) on a standard library type which implements `Index`
    ///
    /// ### Why is this bad?
    /// Using the Index trait (`[]`) is more clear and more
    /// concise.
    ///
    /// ### Known problems
    /// Not a replacement for error handling: Using either
    /// `.unwrap()` or the Index trait (`[]`) carries the risk of causing a `panic`
    /// if the value being accessed is `None`. If the use of `.get().unwrap()` is a
    /// temporary placeholder for dealing with the `Option` type, then this does
    /// not mitigate the need for error handling. If there is a chance that `.get()`
    /// will be `None` in your program, then it is advisable that the `None` case
    /// is handled in a future refactor instead of using `.unwrap()` or the Index
    /// trait.
    ///
    /// ### Example
    /// ```rust
    /// let mut some_vec = vec![0, 1, 2, 3];
    /// let last = some_vec.get(3).unwrap();
    /// *some_vec.get_mut(0).unwrap() = 1;
    /// ```
    /// The correct use would be:
    /// ```rust
    /// let mut some_vec = vec![0, 1, 2, 3];
    /// let last = some_vec[3];
    /// some_vec[0] = 1;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub GET_UNWRAP,
    restriction,
    "using `.get().unwrap()` or `.get_mut().unwrap()` when using `[]` would work instead"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for occurrences where one vector gets extended instead of append
    ///
    /// ### Why is this bad?
    /// Using `append` instead of `extend` is more concise and faster
    ///
    /// ### Example
    /// ```rust
    /// let mut a = vec![1, 2, 3];
    /// let mut b = vec![4, 5, 6];
    ///
    /// a.extend(b.drain(..));
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let mut a = vec![1, 2, 3];
    /// let mut b = vec![4, 5, 6];
    ///
    /// a.append(&mut b);
    /// ```
    #[clippy::version = "1.55.0"]
    pub EXTEND_WITH_DRAIN,
    perf,
    "using vec.append(&mut vec) to move the full range of a vector to another"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the use of `.extend(s.chars())` where s is a
    /// `&str` or `String`.
    ///
    /// ### Why is this bad?
    /// `.push_str(s)` is clearer
    ///
    /// ### Example
    /// ```rust
    /// let abc = "abc";
    /// let def = String::from("def");
    /// let mut s = String::new();
    /// s.extend(abc.chars());
    /// s.extend(def.chars());
    /// ```
    /// The correct use would be:
    /// ```rust
    /// let abc = "abc";
    /// let def = String::from("def");
    /// let mut s = String::new();
    /// s.push_str(abc);
    /// s.push_str(&def);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub STRING_EXTEND_CHARS,
    style,
    "using `x.extend(s.chars())` where s is a `&str` or `String`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the use of `.cloned().collect()` on slice to
    /// create a `Vec`.
    ///
    /// ### Why is this bad?
    /// `.to_vec()` is clearer
    ///
    /// ### Example
    /// ```rust
    /// let s = [1, 2, 3, 4, 5];
    /// let s2: Vec<isize> = s[..].iter().cloned().collect();
    /// ```
    /// The better use would be:
    /// ```rust
    /// let s = [1, 2, 3, 4, 5];
    /// let s2: Vec<isize> = s.to_vec();
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub ITER_CLONED_COLLECT,
    style,
    "using `.cloned().collect()` on slice to create a `Vec`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `_.chars().last()` or
    /// `_.chars().next_back()` on a `str` to check if it ends with a given char.
    ///
    /// ### Why is this bad?
    /// Readability, this can be written more concisely as
    /// `_.ends_with(_)`.
    ///
    /// ### Example
    /// ```rust
    /// # let name = "_";
    /// name.chars().last() == Some('_') || name.chars().next_back() == Some('-');
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # let name = "_";
    /// name.ends_with('_') || name.ends_with('-');
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub CHARS_LAST_CMP,
    style,
    "using `.chars().last()` or `.chars().next_back()` to check if a string ends with a char"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `.as_ref()` or `.as_mut()` where the
    /// types before and after the call are the same.
    ///
    /// ### Why is this bad?
    /// The call is unnecessary.
    ///
    /// ### Example
    /// ```rust
    /// # fn do_stuff(x: &[i32]) {}
    /// let x: &[i32] = &[1, 2, 3, 4, 5];
    /// do_stuff(x.as_ref());
    /// ```
    /// The correct use would be:
    /// ```rust
    /// # fn do_stuff(x: &[i32]) {}
    /// let x: &[i32] = &[1, 2, 3, 4, 5];
    /// do_stuff(x);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub USELESS_ASREF,
    complexity,
    "using `as_ref` where the types before and after the call are the same"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for using `fold` when a more succinct alternative exists.
    /// Specifically, this checks for `fold`s which could be replaced by `any`, `all`,
    /// `sum` or `product`.
    ///
    /// ### Why is this bad?
    /// Readability.
    ///
    /// ### Example
    /// ```rust
    /// # #[allow(unused)]
    /// (0..3).fold(false, |acc, x| acc || x > 2);
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// (0..3).any(|x| x > 2);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub UNNECESSARY_FOLD,
    style,
    "using `fold` when a more succinct alternative exists"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `filter_map` calls that could be replaced by `filter` or `map`.
    /// More specifically it checks if the closure provided is only performing one of the
    /// filter or map operations and suggests the appropriate option.
    ///
    /// ### Why is this bad?
    /// Complexity. The intent is also clearer if only a single
    /// operation is being performed.
    ///
    /// ### Example
    /// ```rust
    /// let _ = (0..3).filter_map(|x| if x > 2 { Some(x) } else { None });
    ///
    /// // As there is no transformation of the argument this could be written as:
    /// let _ = (0..3).filter(|&x| x > 2);
    /// ```
    ///
    /// ```rust
    /// let _ = (0..4).filter_map(|x| Some(x + 1));
    ///
    /// // As there is no conditional check on the argument this could be written as:
    /// let _ = (0..4).map(|x| x + 1);
    /// ```
    #[clippy::version = "1.31.0"]
    pub UNNECESSARY_FILTER_MAP,
    complexity,
    "using `filter_map` when a more succinct alternative exists"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `find_map` calls that could be replaced by `find` or `map`. More
    /// specifically it checks if the closure provided is only performing one of the
    /// find or map operations and suggests the appropriate option.
    ///
    /// ### Why is this bad?
    /// Complexity. The intent is also clearer if only a single
    /// operation is being performed.
    ///
    /// ### Example
    /// ```rust
    /// let _ = (0..3).find_map(|x| if x > 2 { Some(x) } else { None });
    ///
    /// // As there is no transformation of the argument this could be written as:
    /// let _ = (0..3).find(|&x| x > 2);
    /// ```
    ///
    /// ```rust
    /// let _ = (0..4).find_map(|x| Some(x + 1));
    ///
    /// // As there is no conditional check on the argument this could be written as:
    /// let _ = (0..4).map(|x| x + 1).next();
    /// ```
    #[clippy::version = "1.61.0"]
    pub UNNECESSARY_FIND_MAP,
    complexity,
    "using `find_map` when a more succinct alternative exists"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `into_iter` calls on references which should be replaced by `iter`
    /// or `iter_mut`.
    ///
    /// ### Why is this bad?
    /// Readability. Calling `into_iter` on a reference will not move out its
    /// content into the resulting iterator, which is confusing. It is better just call `iter` or
    /// `iter_mut` directly.
    ///
    /// ### Example
    /// ```rust
    /// # let vec = vec![3, 4, 5];
    /// (&vec).into_iter();
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # let vec = vec![3, 4, 5];
    /// (&vec).iter();
    /// ```
    #[clippy::version = "1.32.0"]
    pub INTO_ITER_ON_REF,
    style,
    "using `.into_iter()` on a reference"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to `map` followed by a `count`.
    ///
    /// ### Why is this bad?
    /// It looks suspicious. Maybe `map` was confused with `filter`.
    /// If the `map` call is intentional, this should be rewritten
    /// using `inspect`. Or, if you intend to drive the iterator to
    /// completion, you can just use `for_each` instead.
    ///
    /// ### Example
    /// ```rust
    /// let _ = (0..3).map(|x| x + 2).count();
    /// ```
    #[clippy::version = "1.39.0"]
    pub SUSPICIOUS_MAP,
    suspicious,
    "suspicious usage of map"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `MaybeUninit::uninit().assume_init()`.
    ///
    /// ### Why is this bad?
    /// For most types, this is undefined behavior.
    ///
    /// ### Known problems
    /// For now, we accept empty tuples and tuples / arrays
    /// of `MaybeUninit`. There may be other types that allow uninitialized
    /// data, but those are not yet rigorously defined.
    ///
    /// ### Example
    /// ```rust
    /// // Beware the UB
    /// use std::mem::MaybeUninit;
    ///
    /// let _: usize = unsafe { MaybeUninit::uninit().assume_init() };
    /// ```
    ///
    /// Note that the following is OK:
    ///
    /// ```rust
    /// use std::mem::MaybeUninit;
    ///
    /// let _: [MaybeUninit<bool>; 5] = unsafe {
    ///     MaybeUninit::uninit().assume_init()
    /// };
    /// ```
    #[clippy::version = "1.39.0"]
    pub UNINIT_ASSUMED_INIT,
    correctness,
    "`MaybeUninit::uninit().assume_init()`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `.checked_add/sub(x).unwrap_or(MAX/MIN)`.
    ///
    /// ### Why is this bad?
    /// These can be written simply with `saturating_add/sub` methods.
    ///
    /// ### Example
    /// ```rust
    /// # let y: u32 = 0;
    /// # let x: u32 = 100;
    /// let add = x.checked_add(y).unwrap_or(u32::MAX);
    /// let sub = x.checked_sub(y).unwrap_or(u32::MIN);
    /// ```
    ///
    /// can be written using dedicated methods for saturating addition/subtraction as:
    ///
    /// ```rust
    /// # let y: u32 = 0;
    /// # let x: u32 = 100;
    /// let add = x.saturating_add(y);
    /// let sub = x.saturating_sub(y);
    /// ```
    #[clippy::version = "1.39.0"]
    pub MANUAL_SATURATING_ARITHMETIC,
    style,
    "`.checked_add/sub(x).unwrap_or(MAX/MIN)`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `offset(_)`, `wrapping_`{`add`, `sub`}, etc. on raw pointers to
    /// zero-sized types
    ///
    /// ### Why is this bad?
    /// This is a no-op, and likely unintended
    ///
    /// ### Example
    /// ```rust
    /// unsafe { (&() as *const ()).offset(1) };
    /// ```
    #[clippy::version = "1.41.0"]
    pub ZST_OFFSET,
    correctness,
    "Check for offset calculations on raw pointers to zero-sized types"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `FileType::is_file()`.
    ///
    /// ### Why is this bad?
    /// When people testing a file type with `FileType::is_file`
    /// they are testing whether a path is something they can get bytes from. But
    /// `is_file` doesn't cover special file types in unix-like systems, and doesn't cover
    /// symlink in windows. Using `!FileType::is_dir()` is a better way to that intention.
    ///
    /// ### Example
    /// ```rust
    /// # || {
    /// let metadata = std::fs::metadata("foo.txt")?;
    /// let filetype = metadata.file_type();
    ///
    /// if filetype.is_file() {
    ///     // read file
    /// }
    /// # Ok::<_, std::io::Error>(())
    /// # };
    /// ```
    ///
    /// should be written as:
    ///
    /// ```rust
    /// # || {
    /// let metadata = std::fs::metadata("foo.txt")?;
    /// let filetype = metadata.file_type();
    ///
    /// if !filetype.is_dir() {
    ///     // read file
    /// }
    /// # Ok::<_, std::io::Error>(())
    /// # };
    /// ```
    #[clippy::version = "1.42.0"]
    pub FILETYPE_IS_FILE,
    restriction,
    "`FileType::is_file` is not recommended to test for readable file type"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `_.as_ref().map(Deref::deref)` or its aliases (such as String::as_str).
    ///
    /// ### Why is this bad?
    /// Readability, this can be written more concisely as
    /// `_.as_deref()`.
    ///
    /// ### Example
    /// ```rust
    /// # let opt = Some("".to_string());
    /// opt.as_ref().map(String::as_str)
    /// # ;
    /// ```
    /// Can be written as
    /// ```rust
    /// # let opt = Some("".to_string());
    /// opt.as_deref()
    /// # ;
    /// ```
    #[clippy::version = "1.42.0"]
    pub OPTION_AS_REF_DEREF,
    complexity,
    "using `as_ref().map(Deref::deref)`, which is more succinctly expressed as `as_deref()`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `iter().next()` on a Slice or an Array
    ///
    /// ### Why is this bad?
    /// These can be shortened into `.get()`
    ///
    /// ### Example
    /// ```rust
    /// # let a = [1, 2, 3];
    /// # let b = vec![1, 2, 3];
    /// a[2..].iter().next();
    /// b.iter().next();
    /// ```
    /// should be written as:
    /// ```rust
    /// # let a = [1, 2, 3];
    /// # let b = vec![1, 2, 3];
    /// a.get(2);
    /// b.get(0);
    /// ```
    #[clippy::version = "1.46.0"]
    pub ITER_NEXT_SLICE,
    style,
    "using `.iter().next()` on a sliced array, which can be shortened to just `.get()`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Warns when using `push_str`/`insert_str` with a single-character string literal
    /// where `push`/`insert` with a `char` would work fine.
    ///
    /// ### Why is this bad?
    /// It's less clear that we are pushing a single character.
    ///
    /// ### Example
    /// ```rust
    /// # let mut string = String::new();
    /// string.insert_str(0, "R");
    /// string.push_str("R");
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # let mut string = String::new();
    /// string.insert(0, 'R');
    /// string.push('R');
    /// ```
    #[clippy::version = "1.49.0"]
    pub SINGLE_CHAR_ADD_STR,
    style,
    "`push_str()` or `insert_str()` used with a single-character string literal as parameter"
}

declare_clippy_lint! {
    /// ### What it does
    /// As the counterpart to `or_fun_call`, this lint looks for unnecessary
    /// lazily evaluated closures on `Option` and `Result`.
    ///
    /// This lint suggests changing the following functions, when eager evaluation results in
    /// simpler code:
    ///  - `unwrap_or_else` to `unwrap_or`
    ///  - `and_then` to `and`
    ///  - `or_else` to `or`
    ///  - `get_or_insert_with` to `get_or_insert`
    ///  - `ok_or_else` to `ok_or`
    ///
    /// ### Why is this bad?
    /// Using eager evaluation is shorter and simpler in some cases.
    ///
    /// ### Known problems
    /// It is possible, but not recommended for `Deref` and `Index` to have
    /// side effects. Eagerly evaluating them can change the semantics of the program.
    ///
    /// ### Example
    /// ```rust
    /// // example code where clippy issues a warning
    /// let opt: Option<u32> = None;
    ///
    /// opt.unwrap_or_else(|| 42);
    /// ```
    /// Use instead:
    /// ```rust
    /// let opt: Option<u32> = None;
    ///
    /// opt.unwrap_or(42);
    /// ```
    #[clippy::version = "1.48.0"]
    pub UNNECESSARY_LAZY_EVALUATIONS,
    style,
    "using unnecessary lazy evaluation, which can be replaced with simpler eager evaluation"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `_.map(_).collect::<Result<(), _>()`.
    ///
    /// ### Why is this bad?
    /// Using `try_for_each` instead is more readable and idiomatic.
    ///
    /// ### Example
    /// ```rust
    /// (0..3).map(|t| Err(t)).collect::<Result<(), _>>();
    /// ```
    /// Use instead:
    /// ```rust
    /// (0..3).try_for_each(|t| Err(t));
    /// ```
    #[clippy::version = "1.49.0"]
    pub MAP_COLLECT_RESULT_UNIT,
    style,
    "using `.map(_).collect::<Result<(),_>()`, which can be replaced with `try_for_each`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `from_iter()` function calls on types that implement the `FromIterator`
    /// trait.
    ///
    /// ### Why is this bad?
    /// It is recommended style to use collect. See
    /// [FromIterator documentation](https://doc.rust-lang.org/std/iter/trait.FromIterator.html)
    ///
    /// ### Example
    /// ```rust
    /// let five_fives = std::iter::repeat(5).take(5);
    ///
    /// let v = Vec::from_iter(five_fives);
    ///
    /// assert_eq!(v, vec![5, 5, 5, 5, 5]);
    /// ```
    /// Use instead:
    /// ```rust
    /// let five_fives = std::iter::repeat(5).take(5);
    ///
    /// let v: Vec<i32> = five_fives.collect();
    ///
    /// assert_eq!(v, vec![5, 5, 5, 5, 5]);
    /// ```
    #[clippy::version = "1.49.0"]
    pub FROM_ITER_INSTEAD_OF_COLLECT,
    pedantic,
    "use `.collect()` instead of `::from_iter()`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `inspect().for_each()`.
    ///
    /// ### Why is this bad?
    /// It is the same as performing the computation
    /// inside `inspect` at the beginning of the closure in `for_each`.
    ///
    /// ### Example
    /// ```rust
    /// [1,2,3,4,5].iter()
    /// .inspect(|&x| println!("inspect the number: {}", x))
    /// .for_each(|&x| {
    ///     assert!(x >= 0);
    /// });
    /// ```
    /// Can be written as
    /// ```rust
    /// [1,2,3,4,5].iter()
    /// .for_each(|&x| {
    ///     println!("inspect the number: {}", x);
    ///     assert!(x >= 0);
    /// });
    /// ```
    #[clippy::version = "1.51.0"]
    pub INSPECT_FOR_EACH,
    complexity,
    "using `.inspect().for_each()`, which can be replaced with `.for_each()`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `filter_map(|x| x)`.
    ///
    /// ### Why is this bad?
    /// Readability, this can be written more concisely by using `flatten`.
    ///
    /// ### Example
    /// ```rust
    /// # let iter = vec![Some(1)].into_iter();
    /// iter.filter_map(|x| x);
    /// ```
    /// Use instead:
    /// ```rust
    /// # let iter = vec![Some(1)].into_iter();
    /// iter.flatten();
    /// ```
    #[clippy::version = "1.52.0"]
    pub FILTER_MAP_IDENTITY,
    complexity,
    "call to `filter_map` where `flatten` is sufficient"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for instances of `map(f)` where `f` is the identity function.
    ///
    /// ### Why is this bad?
    /// It can be written more concisely without the call to `map`.
    ///
    /// ### Example
    /// ```rust
    /// let x = [1, 2, 3];
    /// let y: Vec<_> = x.iter().map(|x| x).map(|x| 2*x).collect();
    /// ```
    /// Use instead:
    /// ```rust
    /// let x = [1, 2, 3];
    /// let y: Vec<_> = x.iter().map(|x| 2*x).collect();
    /// ```
    #[clippy::version = "1.47.0"]
    pub MAP_IDENTITY,
    complexity,
    "using iterator.map(|x| x)"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the use of `.bytes().nth()`.
    ///
    /// ### Why is this bad?
    /// `.as_bytes().get()` is more efficient and more
    /// readable.
    ///
    /// ### Example
    /// ```rust
    /// # #[allow(unused)]
    /// "Hello".bytes().nth(3);
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # #[allow(unused)]
    /// "Hello".as_bytes().get(3);
    /// ```
    #[clippy::version = "1.52.0"]
    pub BYTES_NTH,
    style,
    "replace `.bytes().nth()` with `.as_bytes().get()`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the usage of `_.to_owned()`, `vec.to_vec()`, or similar when calling `_.clone()` would be clearer.
    ///
    /// ### Why is this bad?
    /// These methods do the same thing as `_.clone()` but may be confusing as
    /// to why we are calling `to_vec` on something that is already a `Vec` or calling `to_owned` on something that is already owned.
    ///
    /// ### Example
    /// ```rust
    /// let a = vec![1, 2, 3];
    /// let b = a.to_vec();
    /// let c = a.to_owned();
    /// ```
    /// Use instead:
    /// ```rust
    /// let a = vec![1, 2, 3];
    /// let b = a.clone();
    /// let c = a.clone();
    /// ```
    #[clippy::version = "1.52.0"]
    pub IMPLICIT_CLONE,
    pedantic,
    "implicitly cloning a value by invoking a function on its dereferenced type"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the use of `.iter().count()`.
    ///
    /// ### Why is this bad?
    /// `.len()` is more efficient and more
    /// readable.
    ///
    /// ### Example
    /// ```rust
    /// # #![allow(unused)]
    /// let some_vec = vec![0, 1, 2, 3];
    ///
    /// some_vec.iter().count();
    /// &some_vec[..].iter().count();
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let some_vec = vec![0, 1, 2, 3];
    ///
    /// some_vec.len();
    /// &some_vec[..].len();
    /// ```
    #[clippy::version = "1.52.0"]
    pub ITER_COUNT,
    complexity,
    "replace `.iter().count()` with `.len()`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the usage of `_.to_owned()`, on a `Cow<'_, _>`.
    ///
    /// ### Why is this bad?
    /// Calling `to_owned()` on a `Cow` creates a clone of the `Cow`
    /// itself, without taking ownership of the `Cow` contents (i.e.
    /// it's equivalent to calling `Cow::clone`).
    /// The similarly named `into_owned` method, on the other hand,
    /// clones the `Cow` contents, effectively turning any `Cow::Borrowed`
    /// into a `Cow::Owned`.
    ///
    /// Given the potential ambiguity, consider replacing `to_owned`
    /// with `clone` for better readability or, if getting a `Cow::Owned`
    /// was the original intent, using `into_owned` instead.
    ///
    /// ### Example
    /// ```rust
    /// # use std::borrow::Cow;
    /// let s = "Hello world!";
    /// let cow = Cow::Borrowed(s);
    ///
    /// let data = cow.to_owned();
    /// assert!(matches!(data, Cow::Borrowed(_)))
    /// ```
    /// Use instead:
    /// ```rust
    /// # use std::borrow::Cow;
    /// let s = "Hello world!";
    /// let cow = Cow::Borrowed(s);
    ///
    /// let data = cow.clone();
    /// assert!(matches!(data, Cow::Borrowed(_)))
    /// ```
    /// or
    /// ```rust
    /// # use std::borrow::Cow;
    /// let s = "Hello world!";
    /// let cow = Cow::Borrowed(s);
    ///
    /// let _data: String = cow.into_owned();
    /// ```
    #[clippy::version = "1.65.0"]
    pub SUSPICIOUS_TO_OWNED,
    suspicious,
    "calls to `to_owned` on a `Cow<'_, _>` might not do what they are expected"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to [`splitn`]
    /// (https://doc.rust-lang.org/std/primitive.str.html#method.splitn) and
    /// related functions with either zero or one splits.
    ///
    /// ### Why is this bad?
    /// These calls don't actually split the value and are
    /// likely to be intended as a different number.
    ///
    /// ### Example
    /// ```rust
    /// # let s = "";
    /// for x in s.splitn(1, ":") {
    ///     // ..
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # let s = "";
    /// for x in s.splitn(2, ":") {
    ///     // ..
    /// }
    /// ```
    #[clippy::version = "1.54.0"]
    pub SUSPICIOUS_SPLITN,
    correctness,
    "checks for `.splitn(0, ..)` and `.splitn(1, ..)`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for manual implementations of `str::repeat`
    ///
    /// ### Why is this bad?
    /// These are both harder to read, as well as less performant.
    ///
    /// ### Example
    /// ```rust
    /// let x: String = std::iter::repeat('x').take(10).collect();
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let x: String = "x".repeat(10);
    /// ```
    #[clippy::version = "1.54.0"]
    pub MANUAL_STR_REPEAT,
    perf,
    "manual implementation of `str::repeat`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usages of `str::splitn(2, _)`
    ///
    /// ### Why is this bad?
    /// `split_once` is both clearer in intent and slightly more efficient.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let s = "key=value=add";
    /// let (key, value) = s.splitn(2, '=').next_tuple()?;
    /// let value = s.splitn(2, '=').nth(1)?;
    ///
    /// let mut parts = s.splitn(2, '=');
    /// let key = parts.next()?;
    /// let value = parts.next()?;
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// let s = "key=value=add";
    /// let (key, value) = s.split_once('=')?;
    /// let value = s.split_once('=')?.1;
    ///
    /// let (key, value) = s.split_once('=')?;
    /// ```
    ///
    /// ### Limitations
    /// The multiple statement variant currently only detects `iter.next()?`/`iter.next().unwrap()`
    /// in two separate `let` statements that immediately follow the `splitn()`
    #[clippy::version = "1.57.0"]
    pub MANUAL_SPLIT_ONCE,
    complexity,
    "replace `.splitn(2, pat)` with `.split_once(pat)`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usages of `str::splitn` (or `str::rsplitn`) where using `str::split` would be the same.
    /// ### Why is this bad?
    /// The function `split` is simpler and there is no performance difference in these cases, considering
    /// that both functions return a lazy iterator.
    /// ### Example
    /// ```rust
    /// let str = "key=value=add";
    /// let _ = str.splitn(3, '=').next().unwrap();
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let str = "key=value=add";
    /// let _ = str.split('=').next().unwrap();
    /// ```
    #[clippy::version = "1.59.0"]
    pub NEEDLESS_SPLITN,
    complexity,
    "usages of `str::splitn` that can be replaced with `str::split`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for unnecessary calls to [`ToOwned::to_owned`](https://doc.rust-lang.org/std/borrow/trait.ToOwned.html#tymethod.to_owned)
    /// and other `to_owned`-like functions.
    ///
    /// ### Why is this bad?
    /// The unnecessary calls result in useless allocations.
    ///
    /// ### Known problems
    /// `unnecessary_to_owned` can falsely trigger if `IntoIterator::into_iter` is applied to an
    /// owned copy of a resource and the resource is later used mutably. See
    /// [#8148](https://github.com/rust-lang/rust-clippy/issues/8148).
    ///
    /// ### Example
    /// ```rust
    /// let path = std::path::Path::new("x");
    /// foo(&path.to_string_lossy().to_string());
    /// fn foo(s: &str) {}
    /// ```
    /// Use instead:
    /// ```rust
    /// let path = std::path::Path::new("x");
    /// foo(&path.to_string_lossy());
    /// fn foo(s: &str) {}
    /// ```
    #[clippy::version = "1.59.0"]
    pub UNNECESSARY_TO_OWNED,
    perf,
    "unnecessary calls to `to_owned`-like functions"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for use of `.collect::<Vec<String>>().join("")` on iterators.
    ///
    /// ### Why is this bad?
    /// `.collect::<String>()` is more concise and might be more performant
    ///
    /// ### Example
    /// ```rust
    /// let vector = vec!["hello",  "world"];
    /// let output = vector.iter().map(|item| item.to_uppercase()).collect::<Vec<String>>().join("");
    /// println!("{}", output);
    /// ```
    /// The correct use would be:
    /// ```rust
    /// let vector = vec!["hello",  "world"];
    /// let output = vector.iter().map(|item| item.to_uppercase()).collect::<String>();
    /// println!("{}", output);
    /// ```
    /// ### Known problems
    /// While `.collect::<String>()` is sometimes more performant, there are cases where
    /// using `.collect::<String>()` over `.collect::<Vec<String>>().join("")`
    /// will prevent loop unrolling and will result in a negative performance impact.
    ///
    /// Additionally, differences have been observed between aarch64 and x86_64 assembly output,
    /// with aarch64 tending to producing faster assembly in more cases when using `.collect::<String>()`
    #[clippy::version = "1.61.0"]
    pub UNNECESSARY_JOIN,
    pedantic,
    "using `.collect::<Vec<String>>().join(\"\")` on an iterator"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for no-op uses of `Option::{as_deref, as_deref_mut}`,
    /// for example, `Option<&T>::as_deref()` returns the same type.
    ///
    /// ### Why is this bad?
    /// Redundant code and improving readability.
    ///
    /// ### Example
    /// ```rust
    /// let a = Some(&1);
    /// let b = a.as_deref(); // goes from Option<&i32> to Option<&i32>
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let a = Some(&1);
    /// let b = a;
    /// ```
    #[clippy::version = "1.57.0"]
    pub NEEDLESS_OPTION_AS_DEREF,
    complexity,
    "no-op use of `deref` or `deref_mut` method to `Option`."
}

declare_clippy_lint! {
    /// ### What it does
    /// Finds usages of [`char::is_digit`](https://doc.rust-lang.org/stable/std/primitive.char.html#method.is_digit) that
    /// can be replaced with [`is_ascii_digit`](https://doc.rust-lang.org/stable/std/primitive.char.html#method.is_ascii_digit) or
    /// [`is_ascii_hexdigit`](https://doc.rust-lang.org/stable/std/primitive.char.html#method.is_ascii_hexdigit).
    ///
    /// ### Why is this bad?
    /// `is_digit(..)` is slower and requires specifying the radix.
    ///
    /// ### Example
    /// ```rust
    /// let c: char = '6';
    /// c.is_digit(10);
    /// c.is_digit(16);
    /// ```
    /// Use instead:
    /// ```rust
    /// let c: char = '6';
    /// c.is_ascii_digit();
    /// c.is_ascii_hexdigit();
    /// ```
    #[clippy::version = "1.62.0"]
    pub IS_DIGIT_ASCII_RADIX,
    style,
    "use of `char::is_digit(..)` with literal radix of 10 or 16"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calling `take` function after `as_ref`.
    ///
    /// ### Why is this bad?
    /// Redundant code. `take` writes `None` to its argument.
    /// In this case the modification is useless as it's a temporary that cannot be read from afterwards.
    ///
    /// ### Example
    /// ```rust
    /// let x = Some(3);
    /// x.as_ref().take();
    /// ```
    /// Use instead:
    /// ```rust
    /// let x = Some(3);
    /// x.as_ref();
    /// ```
    #[clippy::version = "1.62.0"]
    pub NEEDLESS_OPTION_TAKE,
    complexity,
    "using `.as_ref().take()` on a temporary value"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `replace` statements which have no effect.
    ///
    /// ### Why is this bad?
    /// It's either a mistake or confusing.
    ///
    /// ### Example
    /// ```rust
    /// "1234".replace("12", "12");
    /// "1234".replacen("12", "12", 1);
    /// ```
    #[clippy::version = "1.63.0"]
    pub NO_EFFECT_REPLACE,
    suspicious,
    "replace with no effect"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usages of `.then_some(..).unwrap_or(..)`
    ///
    /// ### Why is this bad?
    /// This can be written more clearly with `if .. else ..`
    ///
    /// ### Limitations
    /// This lint currently only looks for usages of
    /// `.then_some(..).unwrap_or(..)`, but will be expanded
    /// to account for similar patterns.
    ///
    /// ### Example
    /// ```rust
    /// let x = true;
    /// x.then_some("a").unwrap_or("b");
    /// ```
    /// Use instead:
    /// ```rust
    /// let x = true;
    /// if x { "a" } else { "b" };
    /// ```
    #[clippy::version = "1.64.0"]
    pub OBFUSCATED_IF_ELSE,
    style,
    "use of `.then_some(..).unwrap_or(..)` can be written \
    more clearly with `if .. else ..`"
}

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks for calls to `iter`, `iter_mut` or `into_iter` on collections containing a single item
    ///
    /// ### Why is this bad?
    ///
    /// It is simpler to use the once function from the standard library:
    ///
    /// ### Example
    ///
    /// ```rust
    /// let a = [123].iter();
    /// let b = Some(123).into_iter();
    /// ```
    /// Use instead:
    /// ```rust
    /// use std::iter;
    /// let a = iter::once(&123);
    /// let b = iter::once(123);
    /// ```
    ///
    /// ### Known problems
    ///
    /// The type of the resulting iterator might become incompatible with its usage
    #[clippy::version = "1.65.0"]
    pub ITER_ON_SINGLE_ITEMS,
    nursery,
    "Iterator for array of length 1"
}

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks for calls to `iter`, `iter_mut` or `into_iter` on empty collections
    ///
    /// ### Why is this bad?
    ///
    /// It is simpler to use the empty function from the standard library:
    ///
    /// ### Example
    ///
    /// ```rust
    /// use std::{slice, option};
    /// let a: slice::Iter<i32> = [].iter();
    /// let f: option::IntoIter<i32> = None.into_iter();
    /// ```
    /// Use instead:
    /// ```rust
    /// use std::iter;
    /// let a: iter::Empty<i32> = iter::empty();
    /// let b: iter::Empty<i32> = iter::empty();
    /// ```
    ///
    /// ### Known problems
    ///
    /// The type of the resulting iterator might become incompatible with its usage
    #[clippy::version = "1.65.0"]
    pub ITER_ON_EMPTY_COLLECTIONS,
    nursery,
    "Iterator for empty array"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for naive byte counts
    ///
    /// ### Why is this bad?
    /// The [`bytecount`](https://crates.io/crates/bytecount)
    /// crate has methods to count your bytes faster, especially for large slices.
    ///
    /// ### Known problems
    /// If you have predominantly small slices, the
    /// `bytecount::count(..)` method may actually be slower. However, if you can
    /// ensure that less than 2³²-1 matches arise, the `naive_count_32(..)` can be
    /// faster in those cases.
    ///
    /// ### Example
    /// ```rust
    /// # let vec = vec![1_u8];
    /// let count = vec.iter().filter(|x| **x == 0u8).count();
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// # let vec = vec![1_u8];
    /// let count = bytecount::count(&vec, 0u8);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub NAIVE_BYTECOUNT,
    pedantic,
    "use of naive `<slice>.filter(|&x| x == y).count()` to count byte values"
}

declare_clippy_lint! {
    /// ### What it does
    /// It checks for `str::bytes().count()` and suggests replacing it with
    /// `str::len()`.
    ///
    /// ### Why is this bad?
    /// `str::bytes().count()` is longer and may not be as performant as using
    /// `str::len()`.
    ///
    /// ### Example
    /// ```rust
    /// "hello".bytes().count();
    /// String::from("hello").bytes().count();
    /// ```
    /// Use instead:
    /// ```rust
    /// "hello".len();
    /// String::from("hello").len();
    /// ```
    #[clippy::version = "1.62.0"]
    pub BYTES_COUNT_TO_LEN,
    complexity,
    "Using `bytes().count()` when `len()` performs the same functionality"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to `ends_with` with possible file extensions
    /// and suggests to use a case-insensitive approach instead.
    ///
    /// ### Why is this bad?
    /// `ends_with` is case-sensitive and may not detect files with a valid extension.
    ///
    /// ### Example
    /// ```rust
    /// fn is_rust_file(filename: &str) -> bool {
    ///     filename.ends_with(".rs")
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// fn is_rust_file(filename: &str) -> bool {
    ///     let filename = std::path::Path::new(filename);
    ///     filename.extension()
    ///         .map_or(false, |ext| ext.eq_ignore_ascii_case("rs"))
    /// }
    /// ```
    #[clippy::version = "1.51.0"]
    pub CASE_SENSITIVE_FILE_EXTENSION_COMPARISONS,
    pedantic,
    "Checks for calls to ends_with with case-sensitive file extensions"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for using `x.get(0)` instead of
    /// `x.first()`.
    ///
    /// ### Why is this bad?
    /// Using `x.first()` is easier to read and has the same
    /// result.
    ///
    /// ### Example
    /// ```rust
    /// let x = vec![2, 3, 5];
    /// let first_element = x.get(0);
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let x = vec![2, 3, 5];
    /// let first_element = x.first();
    /// ```
    #[clippy::version = "1.63.0"]
    pub GET_FIRST,
    style,
    "Using `x.get(0)` when `x.first()` is simpler"
}

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Finds patterns that reimplement `Option::ok_or`.
    ///
    /// ### Why is this bad?
    ///
    /// Concise code helps focusing on behavior instead of boilerplate.
    ///
    /// ### Examples
    /// ```rust
    /// let foo: Option<i32> = None;
    /// foo.map_or(Err("error"), |v| Ok(v));
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let foo: Option<i32> = None;
    /// foo.ok_or("error");
    /// ```
    #[clippy::version = "1.49.0"]
    pub MANUAL_OK_OR,
    pedantic,
    "finds patterns that can be encoded more concisely with `Option::ok_or`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `map(|x| x.clone())` or
    /// dereferencing closures for `Copy` types, on `Iterator` or `Option`,
    /// and suggests `cloned()` or `copied()` instead
    ///
    /// ### Why is this bad?
    /// Readability, this can be written more concisely
    ///
    /// ### Example
    /// ```rust
    /// let x = vec![42, 43];
    /// let y = x.iter();
    /// let z = y.map(|i| *i);
    /// ```
    ///
    /// The correct use would be:
    ///
    /// ```rust
    /// let x = vec![42, 43];
    /// let y = x.iter();
    /// let z = y.cloned();
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub MAP_CLONE,
    style,
    "using `iterator.map(|x| x.clone())`, or dereferencing closures for `Copy` types"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for instances of `map_err(|_| Some::Enum)`
    ///
    /// ### Why is this bad?
    /// This `map_err` throws away the original error rather than allowing the enum to contain and report the cause of the error
    ///
    /// ### Example
    /// Before:
    /// ```rust
    /// use std::fmt;
    ///
    /// #[derive(Debug)]
    /// enum Error {
    ///     Indivisible,
    ///     Remainder(u8),
    /// }
    ///
    /// impl fmt::Display for Error {
    ///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         match self {
    ///             Error::Indivisible => write!(f, "could not divide input by three"),
    ///             Error::Remainder(remainder) => write!(
    ///                 f,
    ///                 "input is not divisible by three, remainder = {}",
    ///                 remainder
    ///             ),
    ///         }
    ///     }
    /// }
    ///
    /// impl std::error::Error for Error {}
    ///
    /// fn divisible_by_3(input: &str) -> Result<(), Error> {
    ///     input
    ///         .parse::<i32>()
    ///         .map_err(|_| Error::Indivisible)
    ///         .map(|v| v % 3)
    ///         .and_then(|remainder| {
    ///             if remainder == 0 {
    ///                 Ok(())
    ///             } else {
    ///                 Err(Error::Remainder(remainder as u8))
    ///             }
    ///         })
    /// }
    ///  ```
    ///
    ///  After:
    ///  ```rust
    /// use std::{fmt, num::ParseIntError};
    ///
    /// #[derive(Debug)]
    /// enum Error {
    ///     Indivisible(ParseIntError),
    ///     Remainder(u8),
    /// }
    ///
    /// impl fmt::Display for Error {
    ///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         match self {
    ///             Error::Indivisible(_) => write!(f, "could not divide input by three"),
    ///             Error::Remainder(remainder) => write!(
    ///                 f,
    ///                 "input is not divisible by three, remainder = {}",
    ///                 remainder
    ///             ),
    ///         }
    ///     }
    /// }
    ///
    /// impl std::error::Error for Error {
    ///     fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
    ///         match self {
    ///             Error::Indivisible(source) => Some(source),
    ///             _ => None,
    ///         }
    ///     }
    /// }
    ///
    /// fn divisible_by_3(input: &str) -> Result<(), Error> {
    ///     input
    ///         .parse::<i32>()
    ///         .map_err(Error::Indivisible)
    ///         .map(|v| v % 3)
    ///         .and_then(|remainder| {
    ///             if remainder == 0 {
    ///                 Ok(())
    ///             } else {
    ///                 Err(Error::Remainder(remainder as u8))
    ///             }
    ///         })
    /// }
    /// ```
    #[clippy::version = "1.48.0"]
    pub MAP_ERR_IGNORE,
    restriction,
    "`map_err` should not ignore the original error"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `&mut Mutex::lock` calls
    ///
    /// ### Why is this bad?
    /// `Mutex::lock` is less efficient than
    /// calling `Mutex::get_mut`. In addition you also have a statically
    /// guarantee that the mutex isn't locked, instead of just a runtime
    /// guarantee.
    ///
    /// ### Example
    /// ```rust
    /// use std::sync::{Arc, Mutex};
    ///
    /// let mut value_rc = Arc::new(Mutex::new(42_u8));
    /// let value_mutex = Arc::get_mut(&mut value_rc).unwrap();
    ///
    /// let mut value = value_mutex.lock().unwrap();
    /// *value += 1;
    /// ```
    /// Use instead:
    /// ```rust
    /// use std::sync::{Arc, Mutex};
    ///
    /// let mut value_rc = Arc::new(Mutex::new(42_u8));
    /// let value_mutex = Arc::get_mut(&mut value_rc).unwrap();
    ///
    /// let value = value_mutex.get_mut().unwrap();
    /// *value += 1;
    /// ```
    #[clippy::version = "1.49.0"]
    pub MUT_MUTEX_LOCK,
    style,
    "`&mut Mutex::lock` does unnecessary locking"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for duplicate open options as well as combinations
    /// that make no sense.
    ///
    /// ### Why is this bad?
    /// In the best case, the code will be harder to read than
    /// necessary. I don't know the worst case.
    ///
    /// ### Example
    /// ```rust
    /// use std::fs::OpenOptions;
    ///
    /// OpenOptions::new().read(true).truncate(true);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub NONSENSICAL_OPEN_OPTIONS,
    correctness,
    "nonsensical combination of options for opening a file"
}

declare_clippy_lint! {
    /// ### What it does
    ///* Checks for [push](https://doc.rust-lang.org/std/path/struct.PathBuf.html#method.push)
    /// calls on `PathBuf` that can cause overwrites.
    ///
    /// ### Why is this bad?
    /// Calling `push` with a root path at the start can overwrite the
    /// previous defined path.
    ///
    /// ### Example
    /// ```rust
    /// use std::path::PathBuf;
    ///
    /// let mut x = PathBuf::from("/foo");
    /// x.push("/bar");
    /// assert_eq!(x, PathBuf::from("/bar"));
    /// ```
    /// Could be written:
    ///
    /// ```rust
    /// use std::path::PathBuf;
    ///
    /// let mut x = PathBuf::from("/foo");
    /// x.push("bar");
    /// assert_eq!(x, PathBuf::from("/foo/bar"));
    /// ```
    #[clippy::version = "1.36.0"]
    pub PATH_BUF_PUSH_OVERWRITE,
    nursery,
    "calling `push` with file system root on `PathBuf` can overwrite it"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for zipping a collection with the range of
    /// `0.._.len()`.
    ///
    /// ### Why is this bad?
    /// The code is better expressed with `.enumerate()`.
    ///
    /// ### Example
    /// ```rust
    /// # let x = vec![1];
    /// let _ = x.iter().zip(0..x.len());
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # let x = vec![1];
    /// let _ = x.iter().enumerate();
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub RANGE_ZIP_WITH_LEN,
    complexity,
    "zipping iterator with a range when `enumerate()` would do"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `.repeat(1)` and suggest the following method for each types.
    /// - `.to_string()` for `str`
    /// - `.clone()` for `String`
    /// - `.to_vec()` for `slice`
    ///
    /// The lint will evaluate constant expressions and values as arguments of `.repeat(..)` and emit a message if
    /// they are equivalent to `1`. (Related discussion in [rust-clippy#7306](https://github.com/rust-lang/rust-clippy/issues/7306))
    ///
    /// ### Why is this bad?
    /// For example, `String.repeat(1)` is equivalent to `.clone()`. If cloning
    /// the string is the intention behind this, `clone()` should be used.
    ///
    /// ### Example
    /// ```rust
    /// fn main() {
    ///     let x = String::from("hello world").repeat(1);
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// fn main() {
    ///     let x = String::from("hello world").clone();
    /// }
    /// ```
    #[clippy::version = "1.47.0"]
    pub REPEAT_ONCE,
    complexity,
    "using `.repeat(1)` instead of `String.clone()`, `str.to_string()` or `slice.to_vec()` "
}

declare_clippy_lint! {
    /// ### What it does
    /// When sorting primitive values (integers, bools, chars, as well
    /// as arrays, slices, and tuples of such items), it is typically better to
    /// use an unstable sort than a stable sort.
    ///
    /// ### Why is this bad?
    /// Typically, using a stable sort consumes more memory and cpu cycles.
    /// Because values which compare equal are identical, preserving their
    /// relative order (the guarantee that a stable sort provides) means
    /// nothing, while the extra costs still apply.
    ///
    /// ### Known problems
    ///
    /// As pointed out in
    /// [issue #8241](https://github.com/rust-lang/rust-clippy/issues/8241),
    /// a stable sort can instead be significantly faster for certain scenarios
    /// (eg. when a sorted vector is extended with new data and resorted).
    ///
    /// For more information and benchmarking results, please refer to the
    /// issue linked above.
    ///
    /// ### Example
    /// ```rust
    /// let mut vec = vec![2, 1, 3];
    /// vec.sort();
    /// ```
    /// Use instead:
    /// ```rust
    /// let mut vec = vec![2, 1, 3];
    /// vec.sort_unstable();
    /// ```
    #[clippy::version = "1.47.0"]
    pub STABLE_SORT_PRIMITIVE,
    pedantic,
    "use of sort() when sort_unstable() is equivalent"
}

declare_clippy_lint! {
    /// ### What it does
    /// Detects `().hash(_)`.
    ///
    /// ### Why is this bad?
    /// Hashing a unit value doesn't do anything as the implementation of `Hash` for `()` is a no-op.
    ///
    /// ### Example
    /// ```rust
    /// # use std::hash::Hash;
    /// # use std::collections::hash_map::DefaultHasher;
    /// # enum Foo { Empty, WithValue(u8) }
    /// # use Foo::*;
    /// # let mut state = DefaultHasher::new();
    /// # let my_enum = Foo::Empty;
    /// match my_enum {
    /// 	Empty => ().hash(&mut state),
    /// 	WithValue(x) => x.hash(&mut state),
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// # use std::hash::Hash;
    /// # use std::collections::hash_map::DefaultHasher;
    /// # enum Foo { Empty, WithValue(u8) }
    /// # use Foo::*;
    /// # let mut state = DefaultHasher::new();
    /// # let my_enum = Foo::Empty;
    /// match my_enum {
    /// 	Empty => 0_u8.hash(&mut state),
    /// 	WithValue(x) => x.hash(&mut state),
    /// }
    /// ```
    #[clippy::version = "1.58.0"]
    pub UNIT_HASH,
    correctness,
    "hashing a unit value, which does nothing"
}

declare_clippy_lint! {
    /// ### What it does
    /// Detects uses of `Vec::sort_by` passing in a closure
    /// which compares the two arguments, either directly or indirectly.
    ///
    /// ### Why is this bad?
    /// It is more clear to use `Vec::sort_by_key` (or `Vec::sort` if
    /// possible) than to use `Vec::sort_by` and a more complicated
    /// closure.
    ///
    /// ### Known problems
    /// If the suggested `Vec::sort_by_key` uses Reverse and it isn't already
    /// imported by a use statement, then it will need to be added manually.
    ///
    /// ### Example
    /// ```rust
    /// # struct A;
    /// # impl A { fn foo(&self) {} }
    /// # let mut vec: Vec<A> = Vec::new();
    /// vec.sort_by(|a, b| a.foo().cmp(&b.foo()));
    /// ```
    /// Use instead:
    /// ```rust
    /// # struct A;
    /// # impl A { fn foo(&self) {} }
    /// # let mut vec: Vec<A> = Vec::new();
    /// vec.sort_by_key(|a| a.foo());
    /// ```
    #[clippy::version = "1.46.0"]
    pub UNNECESSARY_SORT_BY,
    complexity,
    "Use of `Vec::sort_by` when `Vec::sort_by_key` or `Vec::sort` would be clearer"
}

declare_clippy_lint! {
    /// ### What it does
    /// Finds occurrences of `Vec::resize(0, an_int)`
    ///
    /// ### Why is this bad?
    /// This is probably an argument inversion mistake.
    ///
    /// ### Example
    /// ```rust
    /// vec!(1, 2, 3, 4, 5).resize(0, 5)
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// vec!(1, 2, 3, 4, 5).clear()
    /// ```
    #[clippy::version = "1.46.0"]
    pub VEC_RESIZE_TO_ZERO,
    correctness,
    "emptying a vector with `resize(0, an_int)` instead of `clear()` is probably an argument inversion mistake"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for use of File::read_to_end and File::read_to_string.
    ///
    /// ### Why is this bad?
    /// `fs::{read, read_to_string}` provide the same functionality when `buf` is empty with fewer imports and no intermediate values.
    /// See also: [fs::read docs](https://doc.rust-lang.org/std/fs/fn.read.html), [fs::read_to_string docs](https://doc.rust-lang.org/std/fs/fn.read_to_string.html)
    ///
    /// ### Example
    /// ```rust,no_run
    /// # use std::io::Read;
    /// # use std::fs::File;
    /// let mut f = File::open("foo.txt").unwrap();
    /// let mut bytes = Vec::new();
    /// f.read_to_end(&mut bytes).unwrap();
    /// ```
    /// Can be written more concisely as
    /// ```rust,no_run
    /// # use std::fs;
    /// let mut bytes = fs::read("foo.txt").unwrap();
    /// ```
    #[clippy::version = "1.44.0"]
    pub VERBOSE_FILE_READS,
    restriction,
    "use of `File::read_to_end` or `File::read_to_string`"
}

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks for iterating a map (`HashMap` or `BTreeMap`) and
    /// ignoring either the keys or values.
    ///
    /// ### Why is this bad?
    ///
    /// Readability. There are `keys` and `values` methods that
    /// can be used to express that we only need the keys or the values.
    ///
    /// ### Example
    ///
    /// ```
    /// # use std::collections::HashMap;
    /// let map: HashMap<u32, u32> = HashMap::new();
    /// let values = map.iter().map(|(_, value)| value).collect::<Vec<_>>();
    /// ```
    ///
    /// Use instead:
    /// ```
    /// # use std::collections::HashMap;
    /// let map: HashMap<u32, u32> = HashMap::new();
    /// let values = map.values().collect::<Vec<_>>();
    /// ```
    #[clippy::version = "1.66.0"]
    pub ITER_KV_MAP,
    complexity,
    "iterating on map using `iter` when `keys` or `values` would do"
}

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks an argument of `seek` method of `Seek` trait
    /// and if it start seek from `SeekFrom::Current(0)`, suggests `stream_position` instead.
    ///
    /// ### Why is this bad?
    ///
    /// Readability. Use dedicated method.
    ///
    /// ### Example
    ///
    /// ```rust,no_run
    /// use std::fs::File;
    /// use std::io::{self, Write, Seek, SeekFrom};
    ///
    /// fn main() -> io::Result<()> {
    ///     let mut f = File::create("foo.txt")?;
    ///     f.write_all(b"Hello")?;
    ///     eprintln!("Written {} bytes", f.seek(SeekFrom::Current(0))?);
    ///
    ///     Ok(())
    /// }
    /// ```
    /// Use instead:
    /// ```rust,no_run
    /// use std::fs::File;
    /// use std::io::{self, Write, Seek, SeekFrom};
    ///
    /// fn main() -> io::Result<()> {
    ///     let mut f = File::create("foo.txt")?;
    ///     f.write_all(b"Hello")?;
    ///     eprintln!("Written {} bytes", f.stream_position()?);
    ///
    ///     Ok(())
    /// }
    /// ```
    #[clippy::version = "1.66.0"]
    pub SEEK_FROM_CURRENT,
    complexity,
    "use dedicated method for seek from current position"
}

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks for jumps to the start of a stream that implements `Seek`
    /// and uses the `seek` method providing `Start` as parameter.
    ///
    /// ### Why is this bad?
    ///
    /// Readability. There is a specific method that was implemented for
    /// this exact scenario.
    ///
    /// ### Example
    /// ```rust
    /// # use std::io;
    /// fn foo<T: io::Seek>(t: &mut T) {
    ///     t.seek(io::SeekFrom::Start(0));
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// # use std::io;
    /// fn foo<T: io::Seek>(t: &mut T) {
    ///     t.rewind();
    /// }
    /// ```
    #[clippy::version = "1.66.0"]
    pub SEEK_TO_START_INSTEAD_OF_REWIND,
    complexity,
    "jumping to the start of stream using `seek` method"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for functions collecting an iterator when collect
    /// is not needed.
    ///
    /// ### Why is this bad?
    /// `collect` causes the allocation of a new data structure,
    /// when this allocation may not be needed.
    ///
    /// ### Example
    /// ```rust
    /// # let iterator = vec![1].into_iter();
    /// let len = iterator.clone().collect::<Vec<_>>().len();
    /// // should be
    /// let len = iterator.count();
    /// ```
    #[clippy::version = "1.30.0"]
    pub NEEDLESS_COLLECT,
    nursery,
    "collecting an iterator when collect is not needed"
}

pub struct Methods {
    avoid_breaking_exported_api: bool,
    msrv: Msrv,
    allow_expect_in_tests: bool,
    allow_unwrap_in_tests: bool,
}

impl Methods {
    #[must_use]
    pub fn new(
        avoid_breaking_exported_api: bool,
        msrv: Msrv,
        allow_expect_in_tests: bool,
        allow_unwrap_in_tests: bool,
    ) -> Self {
        Self {
            avoid_breaking_exported_api,
            msrv,
            allow_expect_in_tests,
            allow_unwrap_in_tests,
        }
    }
}

impl_lint_pass!(Methods => [
    UNWRAP_USED,
    EXPECT_USED,
    SHOULD_IMPLEMENT_TRAIT,
    WRONG_SELF_CONVENTION,
    OK_EXPECT,
    UNWRAP_OR_ELSE_DEFAULT,
    MAP_UNWRAP_OR,
    RESULT_MAP_OR_INTO_OPTION,
    OPTION_MAP_OR_NONE,
    BIND_INSTEAD_OF_MAP,
    OR_FUN_CALL,
    OR_THEN_UNWRAP,
    EXPECT_FUN_CALL,
    CHARS_NEXT_CMP,
    CHARS_LAST_CMP,
    CLONE_ON_COPY,
    CLONE_ON_REF_PTR,
    CLONE_DOUBLE_REF,
    COLLAPSIBLE_STR_REPLACE,
    ITER_OVEREAGER_CLONED,
    CLONED_INSTEAD_OF_COPIED,
    FLAT_MAP_OPTION,
    INEFFICIENT_TO_STRING,
    NEW_RET_NO_SELF,
    SINGLE_CHAR_PATTERN,
    SINGLE_CHAR_ADD_STR,
    SEARCH_IS_SOME,
    FILTER_NEXT,
    SKIP_WHILE_NEXT,
    FILTER_MAP_IDENTITY,
    MAP_IDENTITY,
    MANUAL_FILTER_MAP,
    MANUAL_FIND_MAP,
    OPTION_FILTER_MAP,
    FILTER_MAP_NEXT,
    FLAT_MAP_IDENTITY,
    MAP_FLATTEN,
    ITERATOR_STEP_BY_ZERO,
    ITER_NEXT_SLICE,
    ITER_COUNT,
    ITER_NTH,
    ITER_NTH_ZERO,
    BYTES_NTH,
    ITER_SKIP_NEXT,
    GET_UNWRAP,
    GET_LAST_WITH_LEN,
    STRING_EXTEND_CHARS,
    ITER_CLONED_COLLECT,
    ITER_WITH_DRAIN,
    USELESS_ASREF,
    UNNECESSARY_FOLD,
    UNNECESSARY_FILTER_MAP,
    UNNECESSARY_FIND_MAP,
    INTO_ITER_ON_REF,
    SUSPICIOUS_MAP,
    UNINIT_ASSUMED_INIT,
    MANUAL_SATURATING_ARITHMETIC,
    ZST_OFFSET,
    FILETYPE_IS_FILE,
    OPTION_AS_REF_DEREF,
    UNNECESSARY_LAZY_EVALUATIONS,
    MAP_COLLECT_RESULT_UNIT,
    FROM_ITER_INSTEAD_OF_COLLECT,
    INSPECT_FOR_EACH,
    IMPLICIT_CLONE,
    SUSPICIOUS_TO_OWNED,
    SUSPICIOUS_SPLITN,
    MANUAL_STR_REPEAT,
    EXTEND_WITH_DRAIN,
    MANUAL_SPLIT_ONCE,
    NEEDLESS_SPLITN,
    UNNECESSARY_TO_OWNED,
    UNNECESSARY_JOIN,
    ERR_EXPECT,
    NEEDLESS_OPTION_AS_DEREF,
    IS_DIGIT_ASCII_RADIX,
    NEEDLESS_OPTION_TAKE,
    NO_EFFECT_REPLACE,
    OBFUSCATED_IF_ELSE,
    ITER_ON_SINGLE_ITEMS,
    ITER_ON_EMPTY_COLLECTIONS,
    NAIVE_BYTECOUNT,
    BYTES_COUNT_TO_LEN,
    CASE_SENSITIVE_FILE_EXTENSION_COMPARISONS,
    GET_FIRST,
    MANUAL_OK_OR,
    MAP_CLONE,
    MAP_ERR_IGNORE,
    MUT_MUTEX_LOCK,
    NONSENSICAL_OPEN_OPTIONS,
    PATH_BUF_PUSH_OVERWRITE,
    RANGE_ZIP_WITH_LEN,
    REPEAT_ONCE,
    STABLE_SORT_PRIMITIVE,
    UNIT_HASH,
    UNNECESSARY_SORT_BY,
    VEC_RESIZE_TO_ZERO,
    VERBOSE_FILE_READS,
    ITER_KV_MAP,
    SEEK_FROM_CURRENT,
    SEEK_TO_START_INSTEAD_OF_REWIND,
    NEEDLESS_COLLECT,
]);

/// Extracts a method call name, args, and `Span` of the method name.
fn method_call<'tcx>(
    recv: &'tcx hir::Expr<'tcx>,
) -> Option<(&'tcx str, &'tcx hir::Expr<'tcx>, &'tcx [hir::Expr<'tcx>], Span, Span)> {
    if let ExprKind::MethodCall(path, receiver, args, call_span) = recv.kind {
        if !args.iter().any(|e| e.span.from_expansion()) && !receiver.span.from_expansion() {
            let name = path.ident.name.as_str();
            return Some((name, receiver, args, path.ident.span, call_span));
        }
    }
    None
}

impl<'tcx> LateLintPass<'tcx> for Methods {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if expr.span.from_expansion() {
            return;
        }

        self.check_methods(cx, expr);

        match expr.kind {
            hir::ExprKind::Call(func, args) => {
                from_iter_instead_of_collect::check(cx, expr, args, func);
            },
            hir::ExprKind::MethodCall(method_call, receiver, args, _) => {
                let method_span = method_call.ident.span;
                or_fun_call::check(cx, expr, method_span, method_call.ident.as_str(), receiver, args);
                expect_fun_call::check(cx, expr, method_span, method_call.ident.as_str(), receiver, args);
                clone_on_copy::check(cx, expr, method_call.ident.name, receiver, args);
                clone_on_ref_ptr::check(cx, expr, method_call.ident.name, receiver, args);
                inefficient_to_string::check(cx, expr, method_call.ident.name, receiver, args);
                single_char_add_str::check(cx, expr, receiver, args);
                into_iter_on_ref::check(cx, expr, method_span, method_call.ident.name, receiver);
                single_char_pattern::check(cx, expr, method_call.ident.name, receiver, args);
                unnecessary_to_owned::check(cx, expr, method_call.ident.name, receiver, args, &self.msrv);
            },
            hir::ExprKind::Binary(op, lhs, rhs) if op.node == hir::BinOpKind::Eq || op.node == hir::BinOpKind::Ne => {
                let mut info = BinaryExprInfo {
                    expr,
                    chain: lhs,
                    other: rhs,
                    eq: op.node == hir::BinOpKind::Eq,
                };
                lint_binary_expr_with_method_call(cx, &mut info);
            },
            _ => (),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, impl_item: &'tcx hir::ImplItem<'_>) {
        if in_external_macro(cx.sess(), impl_item.span) {
            return;
        }
        let name = impl_item.ident.name.as_str();
        let parent = cx.tcx.hir().get_parent_item(impl_item.hir_id()).def_id;
        let item = cx.tcx.hir().expect_item(parent);
        let self_ty = cx.tcx.type_of(item.owner_id);

        let implements_trait = matches!(item.kind, hir::ItemKind::Impl(hir::Impl { of_trait: Some(_), .. }));
        if let hir::ImplItemKind::Fn(ref sig, id) = impl_item.kind {
            let method_sig = cx.tcx.bound_fn_sig(impl_item.owner_id.to_def_id()).subst_identity();
            let method_sig = cx.tcx.erase_late_bound_regions(method_sig);
            let first_arg_ty_opt = method_sig.inputs().iter().next().copied();
            // if this impl block implements a trait, lint in trait definition instead
            if !implements_trait && cx.effective_visibilities.is_exported(impl_item.owner_id.def_id) {
                // check missing trait implementations
                for method_config in &TRAIT_METHODS {
                    if name == method_config.method_name
                        && sig.decl.inputs.len() == method_config.param_count
                        && method_config.output_type.matches(&sig.decl.output)
                        // in case there is no first arg, since we already have checked the number of arguments
                        // it's should be always true
                        && first_arg_ty_opt.map_or(true, |first_arg_ty| method_config
                            .self_kind.matches(cx, self_ty, first_arg_ty)
                            )
                        && fn_header_equals(method_config.fn_header, sig.header)
                        && method_config.lifetime_param_cond(impl_item)
                    {
                        span_lint_and_help(
                            cx,
                            SHOULD_IMPLEMENT_TRAIT,
                            impl_item.span,
                            &format!(
                                "method `{}` can be confused for the standard trait method `{}::{}`",
                                method_config.method_name, method_config.trait_name, method_config.method_name
                            ),
                            None,
                            &format!(
                                "consider implementing the trait `{}` or choosing a less ambiguous method name",
                                method_config.trait_name
                            ),
                        );
                    }
                }
            }

            if sig.decl.implicit_self.has_implicit_self()
                    && !(self.avoid_breaking_exported_api
                    && cx.effective_visibilities.is_exported(impl_item.owner_id.def_id))
                    && let Some(first_arg) = iter_input_pats(sig.decl, cx.tcx.hir().body(id)).next()
                    && let Some(first_arg_ty) = first_arg_ty_opt
                {
                    wrong_self_convention::check(
                        cx,
                        name,
                        self_ty,
                        first_arg_ty,
                        first_arg.pat.span,
                        implements_trait,
                        false
                    );
                }
        }

        // if this impl block implements a trait, lint in trait definition instead
        if implements_trait {
            return;
        }

        if let hir::ImplItemKind::Fn(_, _) = impl_item.kind {
            let ret_ty = return_ty(cx, impl_item.hir_id());

            if contains_ty_adt_constructor_opaque(cx, ret_ty, self_ty) {
                return;
            }

            if name == "new" && ret_ty != self_ty {
                span_lint(
                    cx,
                    NEW_RET_NO_SELF,
                    impl_item.span,
                    "methods called `new` usually return `Self`",
                );
            }
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx TraitItem<'_>) {
        if in_external_macro(cx.tcx.sess, item.span) {
            return;
        }

        if_chain! {
            if let TraitItemKind::Fn(ref sig, _) = item.kind;
            if sig.decl.implicit_self.has_implicit_self();
            if let Some(first_arg_ty) = sig.decl.inputs.iter().next();

            then {
                let first_arg_span = first_arg_ty.span;
                let first_arg_ty = hir_ty_to_ty(cx.tcx, first_arg_ty);
                let self_ty = TraitRef::identity(cx.tcx, item.owner_id.to_def_id())
                    .self_ty()
                    .skip_binder();
                wrong_self_convention::check(
                    cx,
                    item.ident.name.as_str(),
                    self_ty,
                    first_arg_ty,
                    first_arg_span,
                    false,
                    true,
                );
            }
        }

        if_chain! {
            if item.ident.name == sym::new;
            if let TraitItemKind::Fn(_, _) = item.kind;
            let ret_ty = return_ty(cx, item.hir_id());
            let self_ty = TraitRef::identity(cx.tcx, item.owner_id.to_def_id())
                .self_ty()
                .skip_binder();
            if !ret_ty.contains(self_ty);

            then {
                span_lint(
                    cx,
                    NEW_RET_NO_SELF,
                    item.span,
                    "methods called `new` usually return `Self`",
                );
            }
        }
    }

    extract_msrv_attr!(LateContext);
}

impl Methods {
    #[allow(clippy::too_many_lines)]
    fn check_methods<'tcx>(&self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let Some((name, recv, args, span, call_span)) = method_call(expr) {
            match (name, args) {
                ("add" | "offset" | "sub" | "wrapping_offset" | "wrapping_add" | "wrapping_sub", [_arg]) => {
                    zst_offset::check(cx, expr, recv);
                },
                ("and_then", [arg]) => {
                    let biom_option_linted = bind_instead_of_map::OptionAndThenSome::check(cx, expr, recv, arg);
                    let biom_result_linted = bind_instead_of_map::ResultAndThenOk::check(cx, expr, recv, arg);
                    if !biom_option_linted && !biom_result_linted {
                        unnecessary_lazy_eval::check(cx, expr, recv, arg, "and");
                    }
                },
                ("as_deref" | "as_deref_mut", []) => {
                    needless_option_as_deref::check(cx, expr, recv, name);
                },
                ("as_mut", []) => useless_asref::check(cx, expr, "as_mut", recv),
                ("as_ref", []) => useless_asref::check(cx, expr, "as_ref", recv),
                ("assume_init", []) => uninit_assumed_init::check(cx, expr, recv),
                ("cloned", []) => cloned_instead_of_copied::check(cx, expr, recv, span, &self.msrv),
                ("collect", []) if is_trait_method(cx, expr, sym::Iterator) => {
                    needless_collect::check(cx, span, expr, recv, call_span);
                    match method_call(recv) {
                        Some((name @ ("cloned" | "copied"), recv2, [], _, _)) => {
                            iter_cloned_collect::check(cx, name, expr, recv2);
                        },
                        Some(("map", m_recv, [m_arg], _, _)) => {
                            map_collect_result_unit::check(cx, expr, m_recv, m_arg);
                        },
                        Some(("take", take_self_arg, [take_arg], _, _)) => {
                            if self.msrv.meets(msrvs::STR_REPEAT) {
                                manual_str_repeat::check(cx, expr, recv, take_self_arg, take_arg);
                            }
                        },
                        _ => {},
                    }
                },
                ("count", []) if is_trait_method(cx, expr, sym::Iterator) => match method_call(recv) {
                    Some(("cloned", recv2, [], _, _)) => iter_overeager_cloned::check(cx, expr, recv, recv2, true, false),
                    Some((name2 @ ("into_iter" | "iter" | "iter_mut"), recv2, [], _, _)) => {
                        iter_count::check(cx, expr, recv2, name2);
                    },
                    Some(("map", _, [arg], _, _)) => suspicious_map::check(cx, expr, recv, arg),
                    Some(("filter", recv2, [arg], _, _)) => bytecount::check(cx, expr, recv2, arg),
                    Some(("bytes", recv2, [], _, _)) => bytes_count_to_len::check(cx, expr, recv, recv2),
                    _ => {},
                },
                ("drain", [arg]) => {
                    iter_with_drain::check(cx, expr, recv, span, arg);
                },
                ("ends_with", [arg]) => {
                    if let ExprKind::MethodCall(.., span) = expr.kind {
                        case_sensitive_file_extension_comparisons::check(cx, expr, span, recv, arg);
                    }
                },
                ("expect", [_]) => match method_call(recv) {
                    Some(("ok", recv, [], _, _)) => ok_expect::check(cx, expr, recv),
                    Some(("err", recv, [], err_span, _)) => err_expect::check(cx, expr, recv, span, err_span, &self.msrv),
                    _ => expect_used::check(cx, expr, recv, false, self.allow_expect_in_tests),
                },
                ("expect_err", [_]) => expect_used::check(cx, expr, recv, true, self.allow_expect_in_tests),
                ("extend", [arg]) => {
                    string_extend_chars::check(cx, expr, recv, arg);
                    extend_with_drain::check(cx, expr, recv, arg);
                },
                ("filter_map", [arg]) => {
                    unnecessary_filter_map::check(cx, expr, arg, name);
                    filter_map_identity::check(cx, expr, arg, span);
                },
                ("find_map", [arg]) => {
                    unnecessary_filter_map::check(cx, expr, arg, name);
                },
                ("flat_map", [arg]) => {
                    flat_map_identity::check(cx, expr, arg, span);
                    flat_map_option::check(cx, expr, arg, span);
                },
                ("flatten", []) => match method_call(recv) {
                    Some(("map", recv, [map_arg], map_span, _)) => map_flatten::check(cx, expr, recv, map_arg, map_span),
                    Some(("cloned", recv2, [], _, _)) => iter_overeager_cloned::check(cx, expr, recv, recv2, false, true),
                    _ => {},
                },
                ("fold", [init, acc]) => unnecessary_fold::check(cx, expr, init, acc, span),
                ("for_each", [_]) => {
                    if let Some(("inspect", _, [_], span2, _)) = method_call(recv) {
                        inspect_for_each::check(cx, expr, span2);
                    }
                },
                ("get", [arg]) => {
                    get_first::check(cx, expr, recv, arg);
                    get_last_with_len::check(cx, expr, recv, arg);
                },
                ("get_or_insert_with", [arg]) => unnecessary_lazy_eval::check(cx, expr, recv, arg, "get_or_insert"),
                ("hash", [arg]) => {
                    unit_hash::check(cx, expr, recv, arg);
                },
                ("is_file", []) => filetype_is_file::check(cx, expr, recv),
                ("is_digit", [radix]) => is_digit_ascii_radix::check(cx, expr, recv, radix, &self.msrv),
                ("is_none", []) => check_is_some_is_none(cx, expr, recv, false),
                ("is_some", []) => check_is_some_is_none(cx, expr, recv, true),
                ("iter" | "iter_mut" | "into_iter", []) => {
                    iter_on_single_or_empty_collections::check(cx, expr, name, recv);
                },
                ("join", [join_arg]) => {
                    if let Some(("collect", _, _, span, _)) = method_call(recv) {
                        unnecessary_join::check(cx, expr, recv, join_arg, span);
                    }
                },
                ("last", []) | ("skip", [_]) => {
                    if let Some((name2, recv2, args2, _span2, _)) = method_call(recv) {
                        if let ("cloned", []) = (name2, args2) {
                            iter_overeager_cloned::check(cx, expr, recv, recv2, false, false);
                        }
                    }
                },
                ("lock", []) => {
                    mut_mutex_lock::check(cx, expr, recv, span);
                },
                (name @ ("map" | "map_err"), [m_arg]) => {
                    if name == "map" {
                        map_clone::check(cx, expr, recv, m_arg, &self.msrv);
                        if let Some((map_name @ ("iter" | "into_iter"), recv2, _, _, _)) = method_call(recv) {
                            iter_kv_map::check(cx, map_name, expr, recv2, m_arg);
                        }
                    } else {
                        map_err_ignore::check(cx, expr, m_arg);
                    }
                    if let Some((name, recv2, args, span2,_)) = method_call(recv) {
                        match (name, args) {
                            ("as_mut", []) => option_as_ref_deref::check(cx, expr, recv2, m_arg, true, &self.msrv),
                            ("as_ref", []) => option_as_ref_deref::check(cx, expr, recv2, m_arg, false, &self.msrv),
                            ("filter", [f_arg]) => {
                                filter_map::check(cx, expr, recv2, f_arg, span2, recv, m_arg, span, false);
                            },
                            ("find", [f_arg]) => {
                                filter_map::check(cx, expr, recv2, f_arg, span2, recv, m_arg, span, true);
                            },
                            _ => {},
                        }
                    }
                    map_identity::check(cx, expr, recv, m_arg, name, span);
                },
                ("map_or", [def, map]) => {
                    option_map_or_none::check(cx, expr, recv, def, map);
                    manual_ok_or::check(cx, expr, recv, def, map);
                },
                ("next", []) => {
                    if let Some((name2, recv2, args2, _, _)) = method_call(recv) {
                        match (name2, args2) {
                            ("cloned", []) => iter_overeager_cloned::check(cx, expr, recv, recv2, false, false),
                            ("filter", [arg]) => filter_next::check(cx, expr, recv2, arg),
                            ("filter_map", [arg]) => filter_map_next::check(cx, expr, recv2, arg, &self.msrv),
                            ("iter", []) => iter_next_slice::check(cx, expr, recv2),
                            ("skip", [arg]) => iter_skip_next::check(cx, expr, recv2, arg),
                            ("skip_while", [_]) => skip_while_next::check(cx, expr),
                            _ => {},
                        }
                    }
                },
                ("nth", [n_arg]) => match method_call(recv) {
                    Some(("bytes", recv2, [], _, _)) => bytes_nth::check(cx, expr, recv2, n_arg),
                    Some(("cloned", recv2, [], _, _)) => iter_overeager_cloned::check(cx, expr, recv, recv2, false, false),
                    Some(("iter", recv2, [], _, _)) => iter_nth::check(cx, expr, recv2, recv, n_arg, false),
                    Some(("iter_mut", recv2, [], _, _)) => iter_nth::check(cx, expr, recv2, recv, n_arg, true),
                    _ => iter_nth_zero::check(cx, expr, recv, n_arg),
                },
                ("ok_or_else", [arg]) => unnecessary_lazy_eval::check(cx, expr, recv, arg, "ok_or"),
                ("open", [_]) => {
                    open_options::check(cx, expr, recv);
                },
                ("or_else", [arg]) => {
                    if !bind_instead_of_map::ResultOrElseErrInfo::check(cx, expr, recv, arg) {
                        unnecessary_lazy_eval::check(cx, expr, recv, arg, "or");
                    }
                },
                ("push", [arg]) => {
                    path_buf_push_overwrite::check(cx, expr, arg);
                },
                ("read_to_end", [_]) => {
                    verbose_file_reads::check(cx, expr, recv, verbose_file_reads::READ_TO_END_MSG);
                },
                ("read_to_string", [_]) => {
                    verbose_file_reads::check(cx, expr, recv, verbose_file_reads::READ_TO_STRING_MSG);
                },
                ("repeat", [arg]) => {
                    repeat_once::check(cx, expr, recv, arg);
                },
                (name @ ("replace" | "replacen"), [arg1, arg2] | [arg1, arg2, _]) => {
                    no_effect_replace::check(cx, expr, arg1, arg2);

                    // Check for repeated `str::replace` calls to perform `collapsible_str_replace` lint
                    if self.msrv.meets(msrvs::PATTERN_TRAIT_CHAR_ARRAY)
                        && name == "replace"
                        && let Some(("replace", ..)) = method_call(recv)
                    {
                        collapsible_str_replace::check(cx, expr, arg1, arg2);
                    }
                },
                ("resize", [count_arg, default_arg]) => {
                    vec_resize_to_zero::check(cx, expr, count_arg, default_arg, span);
                },
                ("seek", [arg]) => {
                    if self.msrv.meets(msrvs::SEEK_FROM_CURRENT) {
                        seek_from_current::check(cx, expr, recv, arg);
                    }
                    if self.msrv.meets(msrvs::SEEK_REWIND) {
                        seek_to_start_instead_of_rewind::check(cx, expr, recv, arg, span);
                    }
                },
                ("sort", []) => {
                    stable_sort_primitive::check(cx, expr, recv);
                },
                ("sort_by", [arg]) => {
                    unnecessary_sort_by::check(cx, expr, recv, arg, false);
                },
                ("sort_unstable_by", [arg]) => {
                    unnecessary_sort_by::check(cx, expr, recv, arg, true);
                },
                ("splitn" | "rsplitn", [count_arg, pat_arg]) => {
                    if let Some((Constant::Int(count), _)) = constant(cx, cx.typeck_results(), count_arg) {
                        suspicious_splitn::check(cx, name, expr, recv, count);
                        str_splitn::check(cx, name, expr, recv, pat_arg, count, &self.msrv);
                    }
                },
                ("splitn_mut" | "rsplitn_mut", [count_arg, _]) => {
                    if let Some((Constant::Int(count), _)) = constant(cx, cx.typeck_results(), count_arg) {
                        suspicious_splitn::check(cx, name, expr, recv, count);
                    }
                },
                ("step_by", [arg]) => iterator_step_by_zero::check(cx, expr, arg),
                ("take", [_arg]) => {
                    if let Some((name2, recv2, args2, _span2, _)) = method_call(recv) {
                        if let ("cloned", []) = (name2, args2) {
                            iter_overeager_cloned::check(cx, expr, recv, recv2, false, false);
                        }
                    }
                },
                ("take", []) => needless_option_take::check(cx, expr, recv),
                ("then", [arg]) => {
                    if !self.msrv.meets(msrvs::BOOL_THEN_SOME) {
                        return;
                    }
                    unnecessary_lazy_eval::check(cx, expr, recv, arg, "then_some");
                },
                ("to_owned", []) => {
                    if !suspicious_to_owned::check(cx, expr, recv) {
                        implicit_clone::check(cx, name, expr, recv);
                    }
                },
                ("to_os_string" | "to_path_buf" | "to_vec", []) => {
                    implicit_clone::check(cx, name, expr, recv);
                },
                ("unwrap", []) => {
                    match method_call(recv) {
                        Some(("get", recv, [get_arg], _, _)) => {
                            get_unwrap::check(cx, expr, recv, get_arg, false);
                        },
                        Some(("get_mut", recv, [get_arg], _, _)) => {
                            get_unwrap::check(cx, expr, recv, get_arg, true);
                        },
                        Some(("or", recv, [or_arg], or_span, _)) => {
                            or_then_unwrap::check(cx, expr, recv, or_arg, or_span);
                        },
                        _ => {},
                    }
                    unwrap_used::check(cx, expr, recv, false, self.allow_unwrap_in_tests);
                },
                ("unwrap_err", []) => unwrap_used::check(cx, expr, recv, true, self.allow_unwrap_in_tests),
                ("unwrap_or", [u_arg]) => match method_call(recv) {
                    Some((arith @ ("checked_add" | "checked_sub" | "checked_mul"), lhs, [rhs], _, _)) => {
                        manual_saturating_arithmetic::check(cx, expr, lhs, rhs, u_arg, &arith["checked_".len()..]);
                    },
                    Some(("map", m_recv, [m_arg], span, _)) => {
                        option_map_unwrap_or::check(cx, expr, m_recv, m_arg, recv, u_arg, span);
                    },
                    Some(("then_some", t_recv, [t_arg], _, _)) => {
                        obfuscated_if_else::check(cx, expr, t_recv, t_arg, u_arg);
                    },
                    _ => {},
                },
                ("unwrap_or_else", [u_arg]) => match method_call(recv) {
                    Some(("map", recv, [map_arg], _, _))
                        if map_unwrap_or::check(cx, expr, recv, map_arg, u_arg, &self.msrv) => {},
                    _ => {
                        unwrap_or_else_default::check(cx, expr, recv, u_arg);
                        unnecessary_lazy_eval::check(cx, expr, recv, u_arg, "unwrap_or");
                    },
                },
                ("zip", [arg]) => {
                    if let ExprKind::MethodCall(name, iter_recv, [], _) = recv.kind
                        && name.ident.name == sym::iter
                    {
                        range_zip_with_len::check(cx, expr, iter_recv, arg);
                    }
                },
                _ => {},
            }
        }
    }
}

fn check_is_some_is_none(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, is_some: bool) {
    if let Some((name @ ("find" | "position" | "rposition"), f_recv, [arg], span, _)) = method_call(recv) {
        search_is_some::check(cx, expr, name, is_some, f_recv, arg, recv, span);
    }
}

/// Used for `lint_binary_expr_with_method_call`.
#[derive(Copy, Clone)]
struct BinaryExprInfo<'a> {
    expr: &'a hir::Expr<'a>,
    chain: &'a hir::Expr<'a>,
    other: &'a hir::Expr<'a>,
    eq: bool,
}

/// Checks for the `CHARS_NEXT_CMP` and `CHARS_LAST_CMP` lints.
fn lint_binary_expr_with_method_call(cx: &LateContext<'_>, info: &mut BinaryExprInfo<'_>) {
    macro_rules! lint_with_both_lhs_and_rhs {
        ($func:expr, $cx:expr, $info:ident) => {
            if !$func($cx, $info) {
                ::std::mem::swap(&mut $info.chain, &mut $info.other);
                if $func($cx, $info) {
                    return;
                }
            }
        };
    }

    lint_with_both_lhs_and_rhs!(chars_next_cmp::check, cx, info);
    lint_with_both_lhs_and_rhs!(chars_last_cmp::check, cx, info);
    lint_with_both_lhs_and_rhs!(chars_next_cmp_with_unwrap::check, cx, info);
    lint_with_both_lhs_and_rhs!(chars_last_cmp_with_unwrap::check, cx, info);
}

const FN_HEADER: hir::FnHeader = hir::FnHeader {
    unsafety: hir::Unsafety::Normal,
    constness: hir::Constness::NotConst,
    asyncness: hir::IsAsync::NotAsync,
    abi: rustc_target::spec::abi::Abi::Rust,
};

struct ShouldImplTraitCase {
    trait_name: &'static str,
    method_name: &'static str,
    param_count: usize,
    fn_header: hir::FnHeader,
    // implicit self kind expected (none, self, &self, ...)
    self_kind: SelfKind,
    // checks against the output type
    output_type: OutType,
    // certain methods with explicit lifetimes can't implement the equivalent trait method
    lint_explicit_lifetime: bool,
}
impl ShouldImplTraitCase {
    const fn new(
        trait_name: &'static str,
        method_name: &'static str,
        param_count: usize,
        fn_header: hir::FnHeader,
        self_kind: SelfKind,
        output_type: OutType,
        lint_explicit_lifetime: bool,
    ) -> ShouldImplTraitCase {
        ShouldImplTraitCase {
            trait_name,
            method_name,
            param_count,
            fn_header,
            self_kind,
            output_type,
            lint_explicit_lifetime,
        }
    }

    fn lifetime_param_cond(&self, impl_item: &hir::ImplItem<'_>) -> bool {
        self.lint_explicit_lifetime
            || !impl_item.generics.params.iter().any(|p| {
                matches!(
                    p.kind,
                    hir::GenericParamKind::Lifetime {
                        kind: hir::LifetimeParamKind::Explicit
                    }
                )
            })
    }
}

#[rustfmt::skip]
const TRAIT_METHODS: [ShouldImplTraitCase; 30] = [
    ShouldImplTraitCase::new("std::ops::Add", "add",  2,  FN_HEADER,  SelfKind::Value,  OutType::Any, true),
    ShouldImplTraitCase::new("std::convert::AsMut", "as_mut",  1,  FN_HEADER,  SelfKind::RefMut,  OutType::Ref, true),
    ShouldImplTraitCase::new("std::convert::AsRef", "as_ref",  1,  FN_HEADER,  SelfKind::Ref,  OutType::Ref, true),
    ShouldImplTraitCase::new("std::ops::BitAnd", "bitand",  2,  FN_HEADER,  SelfKind::Value,  OutType::Any, true),
    ShouldImplTraitCase::new("std::ops::BitOr", "bitor",  2,  FN_HEADER,  SelfKind::Value,  OutType::Any, true),
    ShouldImplTraitCase::new("std::ops::BitXor", "bitxor",  2,  FN_HEADER,  SelfKind::Value,  OutType::Any, true),
    ShouldImplTraitCase::new("std::borrow::Borrow", "borrow",  1,  FN_HEADER,  SelfKind::Ref,  OutType::Ref, true),
    ShouldImplTraitCase::new("std::borrow::BorrowMut", "borrow_mut",  1,  FN_HEADER,  SelfKind::RefMut,  OutType::Ref, true),
    ShouldImplTraitCase::new("std::clone::Clone", "clone",  1,  FN_HEADER,  SelfKind::Ref,  OutType::Any, true),
    ShouldImplTraitCase::new("std::cmp::Ord", "cmp",  2,  FN_HEADER,  SelfKind::Ref,  OutType::Any, true),
    ShouldImplTraitCase::new("std::default::Default", "default",  0,  FN_HEADER,  SelfKind::No,  OutType::Any, true),
    ShouldImplTraitCase::new("std::ops::Deref", "deref",  1,  FN_HEADER,  SelfKind::Ref,  OutType::Ref, true),
    ShouldImplTraitCase::new("std::ops::DerefMut", "deref_mut",  1,  FN_HEADER,  SelfKind::RefMut,  OutType::Ref, true),
    ShouldImplTraitCase::new("std::ops::Div", "div",  2,  FN_HEADER,  SelfKind::Value,  OutType::Any, true),
    ShouldImplTraitCase::new("std::ops::Drop", "drop",  1,  FN_HEADER,  SelfKind::RefMut,  OutType::Unit, true),
    ShouldImplTraitCase::new("std::cmp::PartialEq", "eq",  2,  FN_HEADER,  SelfKind::Ref,  OutType::Bool, true),
    ShouldImplTraitCase::new("std::iter::FromIterator", "from_iter",  1,  FN_HEADER,  SelfKind::No,  OutType::Any, true),
    ShouldImplTraitCase::new("std::str::FromStr", "from_str",  1,  FN_HEADER,  SelfKind::No,  OutType::Any, true),
    ShouldImplTraitCase::new("std::hash::Hash", "hash",  2,  FN_HEADER,  SelfKind::Ref,  OutType::Unit, true),
    ShouldImplTraitCase::new("std::ops::Index", "index",  2,  FN_HEADER,  SelfKind::Ref,  OutType::Ref, true),
    ShouldImplTraitCase::new("std::ops::IndexMut", "index_mut",  2,  FN_HEADER,  SelfKind::RefMut,  OutType::Ref, true),
    ShouldImplTraitCase::new("std::iter::IntoIterator", "into_iter",  1,  FN_HEADER,  SelfKind::Value,  OutType::Any, true),
    ShouldImplTraitCase::new("std::ops::Mul", "mul",  2,  FN_HEADER,  SelfKind::Value,  OutType::Any, true),
    ShouldImplTraitCase::new("std::ops::Neg", "neg",  1,  FN_HEADER,  SelfKind::Value,  OutType::Any, true),
    ShouldImplTraitCase::new("std::iter::Iterator", "next",  1,  FN_HEADER,  SelfKind::RefMut,  OutType::Any, false),
    ShouldImplTraitCase::new("std::ops::Not", "not",  1,  FN_HEADER,  SelfKind::Value,  OutType::Any, true),
    ShouldImplTraitCase::new("std::ops::Rem", "rem",  2,  FN_HEADER,  SelfKind::Value,  OutType::Any, true),
    ShouldImplTraitCase::new("std::ops::Shl", "shl",  2,  FN_HEADER,  SelfKind::Value,  OutType::Any, true),
    ShouldImplTraitCase::new("std::ops::Shr", "shr",  2,  FN_HEADER,  SelfKind::Value,  OutType::Any, true),
    ShouldImplTraitCase::new("std::ops::Sub", "sub",  2,  FN_HEADER,  SelfKind::Value,  OutType::Any, true),
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SelfKind {
    Value,
    Ref,
    RefMut,
    No, // When we want the first argument type to be different than `Self`
}

impl SelfKind {
    fn matches<'a>(self, cx: &LateContext<'a>, parent_ty: Ty<'a>, ty: Ty<'a>) -> bool {
        fn matches_value<'a>(cx: &LateContext<'a>, parent_ty: Ty<'a>, ty: Ty<'a>) -> bool {
            if ty == parent_ty {
                true
            } else if ty.is_box() {
                ty.boxed_ty() == parent_ty
            } else if is_type_diagnostic_item(cx, ty, sym::Rc) || is_type_diagnostic_item(cx, ty, sym::Arc) {
                if let ty::Adt(_, substs) = ty.kind() {
                    substs.types().next().map_or(false, |t| t == parent_ty)
                } else {
                    false
                }
            } else {
                false
            }
        }

        fn matches_ref<'a>(cx: &LateContext<'a>, mutability: hir::Mutability, parent_ty: Ty<'a>, ty: Ty<'a>) -> bool {
            if let ty::Ref(_, t, m) = *ty.kind() {
                return m == mutability && t == parent_ty;
            }

            let trait_sym = match mutability {
                hir::Mutability::Not => sym::AsRef,
                hir::Mutability::Mut => sym::AsMut,
            };

            let Some(trait_def_id) = cx.tcx.get_diagnostic_item(trait_sym) else {
                return false
            };
            implements_trait(cx, ty, trait_def_id, &[parent_ty.into()])
        }

        fn matches_none<'a>(cx: &LateContext<'a>, parent_ty: Ty<'a>, ty: Ty<'a>) -> bool {
            !matches_value(cx, parent_ty, ty)
                && !matches_ref(cx, hir::Mutability::Not, parent_ty, ty)
                && !matches_ref(cx, hir::Mutability::Mut, parent_ty, ty)
        }

        match self {
            Self::Value => matches_value(cx, parent_ty, ty),
            Self::Ref => matches_ref(cx, hir::Mutability::Not, parent_ty, ty) || ty == parent_ty && is_copy(cx, ty),
            Self::RefMut => matches_ref(cx, hir::Mutability::Mut, parent_ty, ty),
            Self::No => matches_none(cx, parent_ty, ty),
        }
    }

    #[must_use]
    fn description(self) -> &'static str {
        match self {
            Self::Value => "`self` by value",
            Self::Ref => "`self` by reference",
            Self::RefMut => "`self` by mutable reference",
            Self::No => "no `self`",
        }
    }
}

#[derive(Clone, Copy)]
enum OutType {
    Unit,
    Bool,
    Any,
    Ref,
}

impl OutType {
    fn matches(self, ty: &hir::FnRetTy<'_>) -> bool {
        let is_unit = |ty: &hir::Ty<'_>| matches!(ty.kind, hir::TyKind::Tup(&[]));
        match (self, ty) {
            (Self::Unit, &hir::FnRetTy::DefaultReturn(_)) => true,
            (Self::Unit, &hir::FnRetTy::Return(ty)) if is_unit(ty) => true,
            (Self::Bool, &hir::FnRetTy::Return(ty)) if is_bool(ty) => true,
            (Self::Any, &hir::FnRetTy::Return(ty)) if !is_unit(ty) => true,
            (Self::Ref, &hir::FnRetTy::Return(ty)) => matches!(ty.kind, hir::TyKind::Ref(_, _)),
            _ => false,
        }
    }
}

fn fn_header_equals(expected: hir::FnHeader, actual: hir::FnHeader) -> bool {
    expected.constness == actual.constness
        && expected.unsafety == actual.unsafety
        && expected.asyncness == actual.asyncness
}
