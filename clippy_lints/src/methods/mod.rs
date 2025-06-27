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
mod clear_with_drain;
mod clone_on_copy;
mod clone_on_ref_ptr;
mod cloned_instead_of_copied;
mod collapsible_str_replace;
mod double_ended_iterator_last;
mod drain_collect;
mod err_expect;
mod expect_fun_call;
mod extend_with_drain;
mod filetype_is_file;
mod filter_map;
mod filter_map_bool_then;
mod filter_map_identity;
mod filter_map_next;
mod filter_next;
mod flat_map_identity;
mod flat_map_option;
mod format_collect;
mod from_iter_instead_of_collect;
mod get_first;
mod get_last_with_len;
mod get_unwrap;
mod implicit_clone;
mod inefficient_to_string;
mod inspect_for_each;
mod into_iter_on_ref;
mod io_other_error;
mod ip_constant;
mod is_digit_ascii_radix;
mod is_empty;
mod iter_cloned_collect;
mod iter_count;
mod iter_filter;
mod iter_kv_map;
mod iter_next_slice;
mod iter_nth;
mod iter_nth_zero;
mod iter_on_single_or_empty_collections;
mod iter_out_of_bounds;
mod iter_overeager_cloned;
mod iter_skip_next;
mod iter_skip_zero;
mod iter_with_drain;
mod iterator_step_by_zero;
mod join_absolute_paths;
mod manual_c_str_literals;
mod manual_contains;
mod manual_inspect;
mod manual_is_variant_and;
mod manual_next_back;
mod manual_ok_or;
mod manual_repeat_n;
mod manual_saturating_arithmetic;
mod manual_str_repeat;
mod manual_try_fold;
mod map_all_any_identity;
mod map_clone;
mod map_collect_result_unit;
mod map_err_ignore;
mod map_flatten;
mod map_identity;
mod map_unwrap_or;
mod map_with_unused_argument_over_ranges;
mod mut_mutex_lock;
mod needless_as_bytes;
mod needless_character_iteration;
mod needless_collect;
mod needless_option_as_deref;
mod needless_option_take;
mod no_effect_replace;
mod obfuscated_if_else;
mod ok_expect;
mod open_options;
mod option_as_ref_cloned;
mod option_as_ref_deref;
mod option_map_or_none;
mod option_map_unwrap_or;
mod or_fun_call;
mod or_then_unwrap;
mod path_buf_push_overwrite;
mod path_ends_with_ext;
mod range_zip_with_len;
mod read_line_without_trim;
mod readonly_write_lock;
mod redundant_as_str;
mod repeat_once;
mod result_map_or_else_none;
mod return_and_then;
mod search_is_some;
mod seek_from_current;
mod seek_to_start_instead_of_rewind;
mod single_char_add_str;
mod single_char_insert_string;
mod single_char_push_string;
mod skip_while_next;
mod sliced_string_as_bytes;
mod stable_sort_primitive;
mod str_split;
mod str_splitn;
mod string_extend_chars;
mod string_lit_chars_any;
mod suspicious_command_arg_space;
mod suspicious_map;
mod suspicious_splitn;
mod suspicious_to_owned;
mod swap_with_temporary;
mod type_id_on_box;
mod unbuffered_bytes;
mod uninit_assumed_init;
mod unit_hash;
mod unnecessary_fallible_conversions;
mod unnecessary_filter_map;
mod unnecessary_first_then_check;
mod unnecessary_fold;
mod unnecessary_get_then_check;
mod unnecessary_iter_cloned;
mod unnecessary_join;
mod unnecessary_lazy_eval;
mod unnecessary_literal_unwrap;
mod unnecessary_map_or;
mod unnecessary_min_or_max;
mod unnecessary_result_map_or_else;
mod unnecessary_sort_by;
mod unnecessary_to_owned;
mod unused_enumerate_index;
mod unwrap_expect_used;
mod useless_asref;
mod useless_nonzero_new_unchecked;
mod utils;
mod vec_resize_to_zero;
mod verbose_file_reads;
mod waker_clone_wake;
mod wrong_self_convention;
mod zst_offset;

use clippy_config::Conf;
use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::{span_lint, span_lint_and_help};
use clippy_utils::macros::FormatArgsStorage;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::ty::{contains_ty_adt_constructor_opaque, implements_trait, is_copy, is_type_diagnostic_item};
use clippy_utils::{contains_return, is_bool, is_trait_method, iter_input_pats, peel_blocks, return_ty, sym};
pub use path_ends_with_ext::DEFAULT_ALLOWED_DOTFILES;
use rustc_abi::ExternAbi;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_hir::{Expr, ExprKind, Node, Stmt, StmtKind, TraitItem, TraitItemKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty::{self, TraitRef, Ty};
use rustc_session::impl_lint_pass;
use rustc_span::{Span, Symbol, kw};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `cloned()` on an `Iterator` or `Option` where
    /// `copied()` could be used instead.
    ///
    /// ### Why is this bad?
    /// `copied()` is better because it guarantees that the type being cloned
    /// implements `Copy`.
    ///
    /// ### Example
    /// ```no_run
    /// [1, 2, 3].iter().cloned();
    /// ```
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// let hello = "hesuo worpd"
    ///     .replace('s', "l")
    ///     .replace("u", "l")
    ///     .replace('p', "l");
    /// ```
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// # let vec = vec!["string".to_string()];
    /// vec.iter().cloned().take(10);
    /// vec.iter().cloned().last();
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// Checks for usage of `Iterator::flat_map()` where `filter_map()` could be
    /// used instead.
    ///
    /// ### Why is this bad?
    /// `filter_map()` is known to always produce 0 or 1 output items per input item,
    /// rather than however many the inner iterator type produces.
    /// Therefore, it maintains the upper bound in `Iterator::size_hint()`,
    /// and communicates to the reader that the input items are not being expanded into
    /// multiple output items without their having to notice that the mapping function
    /// returns an `Option`.
    ///
    /// ### Example
    /// ```no_run
    /// let nums: Vec<i32> = ["1", "2", "whee!"].iter().flat_map(|x| x.parse().ok()).collect();
    /// ```
    /// Use instead:
    /// ```no_run
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
    /// ### Why restrict this?
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
    /// ```no_run
    /// # let option = Some(1);
    /// # let result: Result<usize, ()> = Ok(1);
    /// option.unwrap();
    /// result.unwrap();
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// Checks for `.unwrap()` related calls on `Result`s and `Option`s that are constructed.
    ///
    /// ### Why is this bad?
    /// It is better to write the value directly without the indirection.
    ///
    /// ### Examples
    /// ```no_run
    /// let val1 = Some(1).unwrap();
    /// let val2 = Ok::<_, ()>(1).unwrap();
    /// let val3 = Err::<(), _>(1).unwrap_err();
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let val1 = 1;
    /// let val2 = 1;
    /// let val3 = 1;
    /// ```
    #[clippy::version = "1.72.0"]
    pub UNNECESSARY_LITERAL_UNWRAP,
    complexity,
    "using `unwrap()` related calls on `Result` and `Option` constructors"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `.expect()` or `.expect_err()` calls on `Result`s and `.expect()` call on `Option`s.
    ///
    /// ### Why restrict this?
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
    /// ```no_run
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
    /// Checks for methods with certain name prefixes or suffixes, and which
    /// do not adhere to standard conventions regarding how `self` is taken.
    /// The actual rules are:
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
    /// ```no_run
    /// # struct X;
    /// impl X {
    ///     fn as_str(self) -> &'static str {
    ///         // ..
    /// # ""
    ///     }
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # struct X;
    /// impl X {
    ///     fn as_str(&self) -> &'static str {
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
    /// ### Example
    /// ```no_run
    /// # let x = Ok::<_, ()>(());
    /// x.ok().expect("why did I do this again?");
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// Checks for usages of the following functions with an argument that constructs a default value
    /// (e.g., `Default::default` or `String::new`):
    /// - `unwrap_or`
    /// - `unwrap_or_else`
    /// - `or_insert`
    /// - `or_insert_with`
    ///
    /// ### Why is this bad?
    /// Readability. Using `unwrap_or_default` in place of `unwrap_or`/`unwrap_or_else`, or `or_default`
    /// in place of `or_insert`/`or_insert_with`, is simpler and more concise.
    ///
    /// ### Known problems
    /// In some cases, the argument of `unwrap_or`, etc. is needed for type inference. The lint uses a
    /// heuristic to try to identify such cases. However, the heuristic can produce false negatives.
    ///
    /// ### Examples
    /// ```no_run
    /// # let x = Some(1);
    /// # let mut map = std::collections::HashMap::<u64, String>::new();
    /// x.unwrap_or(Default::default());
    /// map.entry(42).or_insert_with(String::new);
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let x = Some(1);
    /// # let mut map = std::collections::HashMap::<u64, String>::new();
    /// x.unwrap_or_default();
    /// map.entry(42).or_default();
    /// ```
    #[clippy::version = "1.56.0"]
    pub UNWRAP_OR_DEFAULT,
    style,
    "using `.unwrap_or`, etc. with an argument that constructs a default value"
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
    /// ```no_run
    /// # let option = Some(1);
    /// # let result: Result<usize, ()> = Ok(1);
    /// # fn some_function(foo: ()) -> usize { 1 }
    /// option.map(|a| a + 1).unwrap_or(0);
    /// option.map(|a| a > 10).unwrap_or(false);
    /// result.map(|a| a + 1).unwrap_or_else(some_function);
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let option = Some(1);
    /// # let result: Result<usize, ()> = Ok(1);
    /// # fn some_function(foo: ()) -> usize { 1 }
    /// option.map_or(0, |a| a + 1);
    /// option.is_some_and(|a| a > 10);
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
    /// ```no_run
    /// # let opt = Some(1);
    /// opt.map_or(None, |a| Some(a + 1));
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// # let r: Result<u32, &str> = Ok(1);
    /// assert_eq!(Some(1), r.map_or(None, Some));
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// Checks for usage of `_.and_then(|x| Some(y))`, `_.and_then(|x| Ok(y))`
    /// or `_.or_else(|x| Err(y))`.
    ///
    /// ### Why is this bad?
    /// This can be written more concisely as `_.map(|x| y)` or `_.map_err(|x| y)`.
    ///
    /// ### Example
    /// ```no_run
    /// # fn opt() -> Option<&'static str> { Some("42") }
    /// # fn res() -> Result<&'static str, &'static str> { Ok("42") }
    /// let _ = opt().and_then(|s| Some(s.len()));
    /// let _ = res().and_then(|s| if s.len() == 42 { Ok(10) } else { Ok(20) });
    /// let _ = res().or_else(|s| if s.len() == 42 { Err(10) } else { Err(20) });
    /// ```
    ///
    /// The correct use would be:
    ///
    /// ```no_run
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
    /// ```no_run
    /// # let vec = vec![1];
    /// vec.iter().filter(|x| **x == 0).next();
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// # let vec = vec![1];
    /// vec.iter().skip_while(|x| **x == 0).next();
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// let vec = vec![vec![1]];
    /// let opt = Some(5);
    ///
    /// vec.iter().map(|x| x.iter()).flatten();
    /// opt.map(|x| Some(x * 2)).flatten();
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// (0_i32..10)
    ///     .filter(|n| n.checked_add(1).is_some())
    ///     .map(|n| n.checked_add(1).unwrap());
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// (0_i32..10)
    ///     .find(|n| n.checked_add(1).is_some())
    ///     .map(|n| n.checked_add(1).unwrap());
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    ///  (0..3).filter_map(|x| if x == 2 { Some(x) } else { None }).next();
    /// ```
    /// Can be written as
    ///
    /// ```no_run
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
    /// ```no_run
    /// # let iter = vec![vec![0]].into_iter();
    /// iter.flat_map(|x| x);
    /// ```
    /// Can be written as
    /// ```no_run
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
    /// ```no_run
    /// let vec = vec![1];
    /// vec.iter().find(|x| **x == 0).is_some();
    ///
    /// "hello world".find("world").is_none();
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let vec = vec![1];
    /// vec.iter().any(|x| *x == 0);
    ///
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
    /// ```no_run
    /// let name = "foo";
    /// if name.chars().next() == Some('_') {};
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// # let foo = Some(String::new());
    /// foo.unwrap_or(String::from("empty"));
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
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
    /// ```no_run
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
    /// ```no_run
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
    /// ```no_run
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
    /// ```no_run
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
    /// ### Why restrict this?
    /// Calling `.clone()` on an `Rc`, `Arc`, or `Weak`
    /// can obscure the fact that only the pointer is being cloned, not the underlying
    /// data.
    ///
    /// ### Example
    /// ```no_run
    /// # use std::rc::Rc;
    /// let x = Rc::new(1);
    ///
    /// x.clone();
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # use std::rc::Rc;
    /// # let x = Rc::new(1);
    /// Rc::clone(&x);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub CLONE_ON_REF_PTR,
    restriction,
    "using `clone` on a ref-counted pointer"
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
    /// ```no_run
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
    /// ```no_run
    /// # struct Foo;
    /// # struct NotAFoo;
    /// impl Foo {
    ///     fn new() -> NotAFoo {
    /// # NotAFoo
    ///     }
    /// }
    /// ```
    ///
    /// ```no_run
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
    /// ```no_run
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
    /// ```no_run
    /// pub trait Trait {
    ///     // Bad. The type name must contain `Self`
    ///     fn new();
    /// }
    /// ```
    ///
    /// ```no_run
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
    /// Checks for iterators of `Option`s using `.filter(Option::is_some).map(Option::unwrap)` that may
    /// be replaced with a `.flatten()` call.
    ///
    /// ### Why is this bad?
    /// `Option` is like a collection of 0-1 things, so `flatten`
    /// automatically does this without suspicious-looking `unwrap` calls.
    ///
    /// ### Example
    /// ```no_run
    /// let _ = std::iter::empty::<Option<i32>>().filter(Option::is_some).map(Option::unwrap);
    /// ```
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// # use std::collections::HashSet;
    /// # let mut s = HashSet::new();
    /// # s.insert(1);
    /// let x = s.iter().nth(0);
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// Checks for usage of `.iter().nth()`/`.iter_mut().nth()` on standard library types that have
    /// equivalent `.get()`/`.get_mut()` methods.
    ///
    /// ### Why is this bad?
    /// `.get()` and `.get_mut()` are equivalent but more concise.
    ///
    /// ### Example
    /// ```no_run
    /// let some_vec = vec![0, 1, 2, 3];
    /// let bad_vec = some_vec.iter().nth(3);
    /// let bad_slice = &some_vec[..].iter().nth(3);
    /// ```
    /// The correct use would be:
    /// ```no_run
    /// let some_vec = vec![0, 1, 2, 3];
    /// let bad_vec = some_vec.get(3);
    /// let bad_slice = &some_vec[..].get(3);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub ITER_NTH,
    style,
    "using `.iter().nth()` on a standard library type with O(1) element access"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `.skip(x).next()` on iterators.
    ///
    /// ### Why is this bad?
    /// `.nth(x)` is cleaner
    ///
    /// ### Example
    /// ```no_run
    /// let some_vec = vec![0, 1, 2, 3];
    /// let bad_vec = some_vec.iter().skip(3).next();
    /// let bad_slice = &some_vec[..].iter().skip(3).next();
    /// ```
    /// The correct use would be:
    /// ```no_run
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
    /// Checks for usage of `.drain(..)` on `Vec` and `VecDeque` for iteration.
    ///
    /// ### Why is this bad?
    /// `.into_iter()` is simpler with better performance.
    ///
    /// ### Example
    /// ```no_run
    /// # use std::collections::HashSet;
    /// let mut foo = vec![0, 1, 2, 3];
    /// let bar: HashSet<usize> = foo.drain(..).collect();
    /// ```
    /// Use instead:
    /// ```no_run
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
    /// Checks for usage of `x.get(x.len() - 1)` instead of
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
    /// ```no_run
    /// let x = vec![2, 3, 5];
    /// let last_element = x.get(x.len() - 1);
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// Checks for usage of `.get().unwrap()` (or
    /// `.get_mut().unwrap`) on a standard library type which implements `Index`
    ///
    /// ### Why restrict this?
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
    /// ```no_run
    /// let mut some_vec = vec![0, 1, 2, 3];
    /// let last = some_vec.get(3).unwrap();
    /// *some_vec.get_mut(0).unwrap() = 1;
    /// ```
    /// The correct use would be:
    /// ```no_run
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
    /// ```no_run
    /// let mut a = vec![1, 2, 3];
    /// let mut b = vec![4, 5, 6];
    ///
    /// a.extend(b.drain(..));
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// let abc = "abc";
    /// let def = String::from("def");
    /// let mut s = String::new();
    /// s.extend(abc.chars());
    /// s.extend(def.chars());
    /// ```
    /// The correct use would be:
    /// ```no_run
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
    /// ```no_run
    /// let s = [1, 2, 3, 4, 5];
    /// let s2: Vec<isize> = s[..].iter().cloned().collect();
    /// ```
    /// The better use would be:
    /// ```no_run
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
    /// ```no_run
    /// # let name = "_";
    /// name.chars().last() == Some('_') || name.chars().next_back() == Some('-');
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// # fn do_stuff(x: &[i32]) {}
    /// let x: &[i32] = &[1, 2, 3, 4, 5];
    /// do_stuff(x.as_ref());
    /// ```
    /// The correct use would be:
    /// ```no_run
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
    /// Checks for usage of `fold` when a more succinct alternative exists.
    /// Specifically, this checks for `fold`s which could be replaced by `any`, `all`,
    /// `sum` or `product`.
    ///
    /// ### Why is this bad?
    /// Readability.
    ///
    /// ### Example
    /// ```no_run
    /// (0..3).fold(false, |acc, x| acc || x > 2);
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// let _ = (0..3).filter_map(|x| if x > 2 { Some(x) } else { None });
    ///
    /// // As there is no transformation of the argument this could be written as:
    /// let _ = (0..3).filter(|&x| x > 2);
    /// ```
    ///
    /// ```no_run
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
    /// ```no_run
    /// let _ = (0..3).find_map(|x| if x > 2 { Some(x) } else { None });
    ///
    /// // As there is no transformation of the argument this could be written as:
    /// let _ = (0..3).find(|&x| x > 2);
    /// ```
    ///
    /// ```no_run
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
    /// ```no_run
    /// # let vec = vec![3, 4, 5];
    /// (&vec).into_iter();
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
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
    /// ```no_run
    /// // Beware the UB
    /// use std::mem::MaybeUninit;
    ///
    /// let _: usize = unsafe { MaybeUninit::uninit().assume_init() };
    /// ```
    ///
    /// Note that the following is OK:
    ///
    /// ```no_run
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
    /// ```no_run
    /// # let y: u32 = 0;
    /// # let x: u32 = 100;
    /// let add = x.checked_add(y).unwrap_or(u32::MAX);
    /// let sub = x.checked_sub(y).unwrap_or(u32::MIN);
    /// ```
    ///
    /// can be written using dedicated methods for saturating addition/subtraction as:
    ///
    /// ```no_run
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
    /// ```no_run
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
    /// ### Why restrict this?
    /// When people testing a file type with `FileType::is_file`
    /// they are testing whether a path is something they can get bytes from. But
    /// `is_file` doesn't cover special file types in unix-like systems, and doesn't cover
    /// symlink in windows. Using `!FileType::is_dir()` is a better way to that intention.
    ///
    /// ### Example
    /// ```no_run
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
    /// ```no_run
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
    /// ```no_run
    /// # let opt = Some("".to_string());
    /// opt.as_ref().map(String::as_str)
    /// # ;
    /// ```
    /// Can be written as
    /// ```no_run
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
    /// ```no_run
    /// # let a = [1, 2, 3];
    /// # let b = vec![1, 2, 3];
    /// a[2..].iter().next();
    /// b.iter().next();
    /// ```
    /// should be written as:
    /// ```no_run
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
    /// ```no_run
    /// # let mut string = String::new();
    /// string.insert_str(0, "R");
    /// string.push_str("R");
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    ///  - `then` to `then_some` (for msrv >= 1.62.0)
    ///
    /// ### Why is this bad?
    /// Using eager evaluation is shorter and simpler in some cases.
    ///
    /// ### Known problems
    /// It is possible, but not recommended for `Deref` and `Index` to have
    /// side effects. Eagerly evaluating them can change the semantics of the program.
    ///
    /// ### Example
    /// ```no_run
    /// let opt: Option<u32> = None;
    ///
    /// opt.unwrap_or_else(|| 42);
    /// ```
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// (0..3).map(|t| Err(t)).collect::<Result<(), _>>();
    /// ```
    /// Use instead:
    /// ```no_run
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
    /// If it's needed to create a collection from the contents of an iterator, the `Iterator::collect(_)`
    /// method is preferred. However, when it's needed to specify the container type,
    /// `Vec::from_iter(_)` can be more readable than using a turbofish (e.g. `_.collect::<Vec<_>>()`). See
    /// [FromIterator documentation](https://doc.rust-lang.org/std/iter/trait.FromIterator.html)
    ///
    /// ### Example
    /// ```no_run
    /// let five_fives = std::iter::repeat(5).take(5);
    ///
    /// let v = Vec::from_iter(five_fives);
    ///
    /// assert_eq!(v, vec![5, 5, 5, 5, 5]);
    /// ```
    /// Use instead:
    /// ```no_run
    /// let five_fives = std::iter::repeat(5).take(5);
    ///
    /// let v: Vec<i32> = five_fives.collect();
    ///
    /// assert_eq!(v, vec![5, 5, 5, 5, 5]);
    /// ```
    /// but prefer to use
    /// ```no_run
    /// let numbers: Vec<i32> = FromIterator::from_iter(1..=5);
    /// ```
    /// instead of
    /// ```no_run
    /// let numbers = (1..=5).collect::<Vec<_>>();
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
    /// ```no_run
    /// [1,2,3,4,5].iter()
    /// .inspect(|&x| println!("inspect the number: {}", x))
    /// .for_each(|&x| {
    ///     assert!(x >= 0);
    /// });
    /// ```
    /// Can be written as
    /// ```no_run
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
    /// ```no_run
    /// # let iter = vec![Some(1)].into_iter();
    /// iter.filter_map(|x| x);
    /// ```
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// let x = [1, 2, 3];
    /// let y: Vec<_> = x.iter().map(|x| x).map(|x| 2*x).collect();
    /// ```
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// "Hello".bytes().nth(3);
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// let a = vec![1, 2, 3];
    /// let b = a.to_vec();
    /// let c = a.to_owned();
    /// ```
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// let some_vec = vec![0, 1, 2, 3];
    ///
    /// some_vec.iter().count();
    /// &some_vec[..].iter().count();
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// # use std::borrow::Cow;
    /// let s = "Hello world!";
    /// let cow = Cow::Borrowed(s);
    ///
    /// let data = cow.to_owned();
    /// assert!(matches!(data, Cow::Borrowed(_)))
    /// ```
    /// Use instead:
    /// ```no_run
    /// # use std::borrow::Cow;
    /// let s = "Hello world!";
    /// let cow = Cow::Borrowed(s);
    ///
    /// let data = cow.clone();
    /// assert!(matches!(data, Cow::Borrowed(_)))
    /// ```
    /// or
    /// ```no_run
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
    /// ```no_run
    /// # let s = "";
    /// for x in s.splitn(1, ":") {
    ///     // ..
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// let x: String = std::iter::repeat('x').take(10).collect();
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let x: String = "x".repeat(10);
    /// ```
    #[clippy::version = "1.54.0"]
    pub MANUAL_STR_REPEAT,
    perf,
    "manual implementation of `str::repeat`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `str::splitn(2, _)`
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
    /// Checks for usage of `str::splitn` (or `str::rsplitn`) where using `str::split` would be the same.
    /// ### Why is this bad?
    /// The function `split` is simpler and there is no performance difference in these cases, considering
    /// that both functions return a lazy iterator.
    /// ### Example
    /// ```no_run
    /// let str = "key=value=add";
    /// let _ = str.splitn(3, '=').next().unwrap();
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// let path = std::path::Path::new("x");
    /// foo(&path.to_string_lossy().to_string());
    /// fn foo(s: &str) {}
    /// ```
    /// Use instead:
    /// ```no_run
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
    /// Checks for usage of `.collect::<Vec<String>>().join("")` on iterators.
    ///
    /// ### Why is this bad?
    /// `.collect::<String>()` is more concise and might be more performant
    ///
    /// ### Example
    /// ```no_run
    /// let vector = vec!["hello",  "world"];
    /// let output = vector.iter().map(|item| item.to_uppercase()).collect::<Vec<String>>().join("");
    /// println!("{}", output);
    /// ```
    /// The correct use would be:
    /// ```no_run
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
    /// ```no_run
    /// let a = Some(&1);
    /// let b = a.as_deref(); // goes from Option<&i32> to Option<&i32>
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// let c: char = '6';
    /// c.is_digit(10);
    /// c.is_digit(16);
    /// ```
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// let x = Some(3);
    /// x.as_ref().take();
    /// ```
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
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
    /// Checks for unnecessary method chains that can be simplified into `if .. else ..`.
    ///
    /// ### Why is this bad?
    /// This can be written more clearly with `if .. else ..`
    ///
    /// ### Limitations
    /// This lint currently only looks for usages of
    /// `.{then, then_some}(..).{unwrap_or, unwrap_or_else, unwrap_or_default}(..)`, but will be expanded
    /// to account for similar patterns.
    ///
    /// ### Example
    /// ```no_run
    /// let x = true;
    /// x.then_some("a").unwrap_or("b");
    /// ```
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// let a = [123].iter();
    /// let b = Some(123).into_iter();
    /// ```
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// use std::{slice, option};
    /// let a: slice::Iter<i32> = [].iter();
    /// let f: option::IntoIter<i32> = None.into_iter();
    /// ```
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
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
    /// ```no_run
    /// "hello".bytes().count();
    /// String::from("hello").bytes().count();
    /// ```
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// fn is_rust_file(filename: &str) -> bool {
    ///     filename.ends_with(".rs")
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
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
    /// Checks for usage of `x.get(0)` instead of
    /// `x.first()` or `x.front()`.
    ///
    /// ### Why is this bad?
    /// Using `x.first()` for `Vec`s and slices or `x.front()`
    /// for `VecDeque`s is easier to read and has the same result.
    ///
    /// ### Example
    /// ```no_run
    /// let x = vec![2, 3, 5];
    /// let first_element = x.get(0);
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let x = vec![2, 3, 5];
    /// let first_element = x.first();
    /// ```
    #[clippy::version = "1.63.0"]
    pub GET_FIRST,
    style,
    "Using `x.get(0)` when `x.first()` or `x.front()` is simpler"
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
    /// ```no_run
    /// let foo: Option<i32> = None;
    /// foo.map_or(Err("error"), |v| Ok(v));
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let foo: Option<i32> = None;
    /// foo.ok_or("error");
    /// ```
    #[clippy::version = "1.49.0"]
    pub MANUAL_OK_OR,
    style,
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
    /// ```no_run
    /// let x = vec![42, 43];
    /// let y = x.iter();
    /// let z = y.map(|i| *i);
    /// ```
    ///
    /// The correct use would be:
    ///
    /// ```no_run
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
    /// ### Why restrict this?
    /// This `map_err` throws away the original error rather than allowing the enum to
    /// contain and report the cause of the error.
    ///
    /// ### Example
    /// Before:
    /// ```no_run
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
    /// ```
    ///
    /// After:
    /// ```rust
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
    /// ```no_run
    /// use std::sync::{Arc, Mutex};
    ///
    /// let mut value_rc = Arc::new(Mutex::new(42_u8));
    /// let value_mutex = Arc::get_mut(&mut value_rc).unwrap();
    ///
    /// let mut value = value_mutex.lock().unwrap();
    /// *value += 1;
    /// ```
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
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
    /// Checks for the suspicious use of `OpenOptions::create()`
    /// without an explicit `OpenOptions::truncate()`.
    ///
    /// ### Why is this bad?
    /// `create()` alone will either create a new file or open an
    /// existing file. If the file already exists, it will be
    /// overwritten when written to, but the file will not be
    /// truncated by default.
    /// If less data is written to the file
    /// than it already contains, the remainder of the file will
    /// remain unchanged, and the end of the file will contain old
    /// data.
    /// In most cases, one should either use `create_new` to ensure
    /// the file is created from scratch, or ensure `truncate` is
    /// called so that the truncation behaviour is explicit. `truncate(true)`
    /// will ensure the file is entirely overwritten with new data, whereas
    /// `truncate(false)` will explicitly keep the default behavior.
    ///
    /// ### Example
    /// ```rust,no_run
    /// use std::fs::OpenOptions;
    ///
    /// OpenOptions::new().create(true);
    /// ```
    /// Use instead:
    /// ```rust,no_run
    /// use std::fs::OpenOptions;
    ///
    /// OpenOptions::new().create(true).truncate(true);
    /// ```
    #[clippy::version = "1.77.0"]
    pub SUSPICIOUS_OPEN_OPTIONS,
    suspicious,
    "suspicious combination of options for opening a file"
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
    /// ```no_run
    /// use std::path::PathBuf;
    ///
    /// let mut x = PathBuf::from("/foo");
    /// x.push("/bar");
    /// assert_eq!(x, PathBuf::from("/bar"));
    /// ```
    /// Could be written:
    ///
    /// ```no_run
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
    /// ```no_run
    /// # let x = vec![1];
    /// let _ = x.iter().zip(0..x.len());
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// fn main() {
    ///     let x = String::from("hello world").repeat(1);
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// let mut vec = vec![2, 1, 3];
    /// vec.sort();
    /// ```
    /// Use instead:
    /// ```no_run
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
    /// Looks for calls to `.type_id()` on a `Box<dyn _>`.
    ///
    /// ### Why is this bad?
    /// This almost certainly does not do what the user expects and can lead to subtle bugs.
    /// Calling `.type_id()` on a `Box<dyn Trait>` returns a fixed `TypeId` of the `Box` itself,
    /// rather than returning the `TypeId` of the underlying type behind the trait object.
    ///
    /// For `Box<dyn Any>` specifically (and trait objects that have `Any` as its supertrait),
    /// this lint will provide a suggestion, which is to dereference the receiver explicitly
    /// to go from `Box<dyn Any>` to `dyn Any`.
    /// This makes sure that `.type_id()` resolves to a dynamic call on the trait object
    /// and not on the box.
    ///
    /// If the fixed `TypeId` of the `Box` is the intended behavior, it's better to be explicit about it
    /// and write `TypeId::of::<Box<dyn Trait>>()`:
    /// this makes it clear that a fixed `TypeId` is returned and not the `TypeId` of the implementor.
    ///
    /// ### Example
    /// ```rust,ignore
    /// use std::any::{Any, TypeId};
    ///
    /// let any_box: Box<dyn Any> = Box::new(42_i32);
    /// assert_eq!(any_box.type_id(), TypeId::of::<i32>()); // ⚠️ this fails!
    /// ```
    /// Use instead:
    /// ```no_run
    /// use std::any::{Any, TypeId};
    ///
    /// let any_box: Box<dyn Any> = Box::new(42_i32);
    /// assert_eq!((*any_box).type_id(), TypeId::of::<i32>());
    /// //          ^ dereference first, to call `type_id` on `dyn Any`
    /// ```
    #[clippy::version = "1.73.0"]
    pub TYPE_ID_ON_BOX,
    suspicious,
    "calling `.type_id()` on a boxed trait object"
}

declare_clippy_lint! {
    /// ### What it does
    /// Detects `().hash(_)`.
    ///
    /// ### Why is this bad?
    /// Hashing a unit value doesn't do anything as the implementation of `Hash` for `()` is a no-op.
    ///
    /// ### Example
    /// ```no_run
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
    /// ```no_run
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
    /// Checks for usage of `Vec::sort_by` passing in a closure
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
    /// ```no_run
    /// # struct A;
    /// # impl A { fn foo(&self) {} }
    /// # let mut vec: Vec<A> = Vec::new();
    /// vec.sort_by(|a, b| a.foo().cmp(&b.foo()));
    /// ```
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// vec![1, 2, 3, 4, 5].resize(0, 5)
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// vec![1, 2, 3, 4, 5].clear()
    /// ```
    #[clippy::version = "1.46.0"]
    pub VEC_RESIZE_TO_ZERO,
    correctness,
    "emptying a vector with `resize(0, an_int)` instead of `clear()` is probably an argument inversion mistake"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of File::read_to_end and File::read_to_string.
    ///
    /// ### Why restrict this?
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
    /// ```no_run
    /// # use std::collections::HashMap;
    /// let map: HashMap<u32, u32> = HashMap::new();
    /// let values = map.iter().map(|(_, value)| value).collect::<Vec<_>>();
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// Checks if the `seek` method of the `Seek` trait is called with `SeekFrom::Current(0)`,
    /// and if it is, suggests using `stream_position` instead.
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
    #[clippy::version = "1.67.0"]
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
    /// ```no_run
    /// # use std::io;
    /// fn foo<T: io::Seek>(t: &mut T) {
    ///     t.seek(io::SeekFrom::Start(0));
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// # use std::io;
    /// fn foo<T: io::Seek>(t: &mut T) {
    ///     t.rewind();
    /// }
    /// ```
    #[clippy::version = "1.67.0"]
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
    /// ```no_run
    /// # let iterator = vec![1].into_iter();
    /// let len = iterator.collect::<Vec<_>>().len();
    /// ```
    /// Use instead:
    /// ```no_run
    /// # let iterator = vec![1].into_iter();
    /// let len = iterator.count();
    /// ```
    #[clippy::version = "1.30.0"]
    pub NEEDLESS_COLLECT,
    nursery,
    "collecting an iterator when collect is not needed"
}

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks for `Command::arg()` invocations that look like they
    /// should be multiple arguments instead, such as `arg("-t ext2")`.
    ///
    /// ### Why is this bad?
    ///
    /// `Command::arg()` does not split arguments by space. An argument like `arg("-t ext2")`
    /// will be passed as a single argument to the command,
    /// which is likely not what was intended.
    ///
    /// ### Example
    /// ```no_run
    /// std::process::Command::new("echo").arg("-n hello").spawn().unwrap();
    /// ```
    /// Use instead:
    /// ```no_run
    /// std::process::Command::new("echo").args(["-n", "hello"]).spawn().unwrap();
    /// ```
    #[clippy::version = "1.69.0"]
    pub SUSPICIOUS_COMMAND_ARG_SPACE,
    suspicious,
    "single command line argument that looks like it should be multiple arguments"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `.drain(..)` for the sole purpose of clearing a container.
    ///
    /// ### Why is this bad?
    /// This creates an unnecessary iterator that is dropped immediately.
    ///
    /// Calling `.clear()` also makes the intent clearer.
    ///
    /// ### Example
    /// ```no_run
    /// let mut v = vec![1, 2, 3];
    /// v.drain(..);
    /// ```
    /// Use instead:
    /// ```no_run
    /// let mut v = vec![1, 2, 3];
    /// v.clear();
    /// ```
    #[clippy::version = "1.70.0"]
    pub CLEAR_WITH_DRAIN,
    nursery,
    "calling `drain` in order to `clear` a container"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `.rev().next()` on a `DoubleEndedIterator`
    ///
    /// ### Why is this bad?
    /// `.next_back()` is cleaner.
    ///
    /// ### Example
    /// ```no_run
    /// # let foo = [0; 10];
    /// foo.iter().rev().next();
    /// ```
    /// Use instead:
    /// ```no_run
    /// # let foo = [0; 10];
    /// foo.iter().next_back();
    /// ```
    #[clippy::version = "1.71.0"]
    pub MANUAL_NEXT_BACK,
    style,
    "manual reverse iteration of `DoubleEndedIterator`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to `.drain()` that clear the collection, immediately followed by a call to `.collect()`.
    ///
    /// > "Collection" in this context refers to any type with a `drain` method:
    /// > `Vec`, `VecDeque`, `BinaryHeap`, `HashSet`,`HashMap`, `String`
    ///
    /// ### Why is this bad?
    /// Using `mem::take` is faster as it avoids the allocation.
    /// When using `mem::take`, the old collection is replaced with an empty one and ownership of
    /// the old collection is returned.
    ///
    /// ### Known issues
    /// `mem::take(&mut vec)` is almost equivalent to `vec.drain(..).collect()`, except that
    /// it also moves the **capacity**. The user might have explicitly written it this way
    /// to keep the capacity on the original `Vec`.
    ///
    /// ### Example
    /// ```no_run
    /// fn remove_all(v: &mut Vec<i32>) -> Vec<i32> {
    ///     v.drain(..).collect()
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// use std::mem;
    /// fn remove_all(v: &mut Vec<i32>) -> Vec<i32> {
    ///     mem::take(v)
    /// }
    /// ```
    #[clippy::version = "1.72.0"]
    pub DRAIN_COLLECT,
    perf,
    "calling `.drain(..).collect()` to move all elements into a new collection"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `Iterator::fold` with a type that implements `Try`.
    ///
    /// ### Why is this bad?
    /// The code should use `try_fold` instead, which short-circuits on failure, thus opening the
    /// door for additional optimizations not possible with `fold` as rustc can guarantee the
    /// function is never called on `None`, `Err`, etc., alleviating otherwise necessary checks. It's
    /// also slightly more idiomatic.
    ///
    /// ### Known issues
    /// This lint doesn't take into account whether a function does something on the failure case,
    /// i.e., whether short-circuiting will affect behavior. Refactoring to `try_fold` is not
    /// desirable in those cases.
    ///
    /// ### Example
    /// ```no_run
    /// vec![1, 2, 3].iter().fold(Some(0i32), |sum, i| sum?.checked_add(*i));
    /// ```
    /// Use instead:
    /// ```no_run
    /// vec![1, 2, 3].iter().try_fold(0i32, |sum, i| sum.checked_add(*i));
    /// ```
    #[clippy::version = "1.72.0"]
    pub MANUAL_TRY_FOLD,
    perf,
    "checks for usage of `Iterator::fold` with a type that implements `Try`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Looks for calls to [`Stdin::read_line`] to read a line from the standard input
    /// into a string, then later attempting to use that string for an operation that will never
    /// work for strings with a trailing newline character in it (e.g. parsing into a `i32`).
    ///
    /// ### Why is this bad?
    /// The operation will always fail at runtime no matter what the user enters, thus
    /// making it a useless operation.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let mut input = String::new();
    /// std::io::stdin().read_line(&mut input).expect("Failed to read a line");
    /// let num: i32 = input.parse().expect("Not a number!");
    /// assert_eq!(num, 42); // we never even get here!
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// let mut input = String::new();
    /// std::io::stdin().read_line(&mut input).expect("Failed to read a line");
    /// let num: i32 = input.trim_end().parse().expect("Not a number!");
    /// //                  ^^^^^^^^^^^ remove the trailing newline
    /// assert_eq!(num, 42);
    /// ```
    #[clippy::version = "1.73.0"]
    pub READ_LINE_WITHOUT_TRIM,
    correctness,
    "calling `Stdin::read_line`, then trying to parse it without first trimming"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `<string_lit>.chars().any(|i| i == c)`.
    ///
    /// ### Why is this bad?
    /// It's significantly slower than using a pattern instead, like
    /// `matches!(c, '\\' | '.' | '+')`.
    ///
    /// Despite this being faster, this is not `perf` as this is pretty common, and is a rather nice
    /// way to check if a `char` is any in a set. In any case, this `restriction` lint is available
    /// for situations where that additional performance is absolutely necessary.
    ///
    /// ### Example
    /// ```no_run
    /// # let c = 'c';
    /// "\\.+*?()|[]{}^$#&-~".chars().any(|x| x == c);
    /// ```
    /// Use instead:
    /// ```no_run
    /// # let c = 'c';
    /// matches!(c, '\\' | '.' | '+' | '*' | '(' | ')' | '|' | '[' | ']' | '{' | '}' | '^' | '$' | '#' | '&' | '-' | '~');
    /// ```
    #[clippy::version = "1.73.0"]
    pub STRING_LIT_CHARS_ANY,
    restriction,
    "checks for `<string_lit>.chars().any(|i| i == c)`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `.map(|_| format!(..)).collect::<String>()`.
    ///
    /// ### Why is this bad?
    /// This allocates a new string for every element in the iterator.
    /// This can be done more efficiently by creating the `String` once and appending to it in `Iterator::fold`,
    /// using either the `write!` macro which supports exactly the same syntax as the `format!` macro,
    /// or concatenating with `+` in case the iterator yields `&str`/`String`.
    ///
    /// Note also that `write!`-ing into a `String` can never fail, despite the return type of `write!` being `std::fmt::Result`,
    /// so it can be safely ignored or unwrapped.
    ///
    /// ### Example
    /// ```no_run
    /// fn hex_encode(bytes: &[u8]) -> String {
    ///     bytes.iter().map(|b| format!("{b:02X}")).collect()
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// use std::fmt::Write;
    /// fn hex_encode(bytes: &[u8]) -> String {
    ///     bytes.iter().fold(String::new(), |mut output, b| {
    ///         let _ = write!(output, "{b:02X}");
    ///         output
    ///     })
    /// }
    /// ```
    #[clippy::version = "1.73.0"]
    pub FORMAT_COLLECT,
    pedantic,
    "`format!`ing every element in a collection, then collecting the strings into a new `String`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `.skip(0)` on iterators.
    ///
    /// ### Why is this bad?
    /// This was likely intended to be `.skip(1)` to skip the first element, as `.skip(0)` does
    /// nothing. If not, the call should be removed.
    ///
    /// ### Example
    /// ```no_run
    /// let v = vec![1, 2, 3];
    /// let x = v.iter().skip(0).collect::<Vec<_>>();
    /// let y = v.iter().collect::<Vec<_>>();
    /// assert_eq!(x, y);
    /// ```
    #[clippy::version = "1.73.0"]
    pub ITER_SKIP_ZERO,
    correctness,
    "disallows `.skip(0)`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `bool::then` in `Iterator::filter_map`.
    ///
    /// ### Why is this bad?
    /// This can be written with `filter` then `map` instead, which would reduce nesting and
    /// separates the filtering from the transformation phase. This comes with no cost to
    /// performance and is just cleaner.
    ///
    /// ### Limitations
    /// Does not lint `bool::then_some`, as it eagerly evaluates its arguments rather than lazily.
    /// This can create differing behavior, so better safe than sorry.
    ///
    /// ### Example
    /// ```no_run
    /// # fn really_expensive_fn(i: i32) -> i32 { i }
    /// # let v = vec![];
    /// _ = v.into_iter().filter_map(|i| (i % 2 == 0).then(|| really_expensive_fn(i)));
    /// ```
    /// Use instead:
    /// ```no_run
    /// # fn really_expensive_fn(i: i32) -> i32 { i }
    /// # let v = vec![];
    /// _ = v.into_iter().filter(|i| i % 2 == 0).map(|i| really_expensive_fn(i));
    /// ```
    #[clippy::version = "1.73.0"]
    pub FILTER_MAP_BOOL_THEN,
    style,
    "checks for usage of `bool::then` in `Iterator::filter_map`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Looks for calls to `RwLock::write` where the lock is only used for reading.
    ///
    /// ### Why is this bad?
    /// The write portion of `RwLock` is exclusive, meaning that no other thread
    /// can access the lock while this writer is active.
    ///
    /// ### Example
    /// ```no_run
    /// use std::sync::RwLock;
    /// fn assert_is_zero(lock: &RwLock<i32>) {
    ///     let num = lock.write().unwrap();
    ///     assert_eq!(*num, 0);
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// use std::sync::RwLock;
    /// fn assert_is_zero(lock: &RwLock<i32>) {
    ///     let num = lock.read().unwrap();
    ///     assert_eq!(*num, 0);
    /// }
    /// ```
    #[clippy::version = "1.73.0"]
    pub READONLY_WRITE_LOCK,
    perf,
    "acquiring a write lock when a read lock would work"
}

declare_clippy_lint! {
    /// ### What it does
    /// Looks for iterator combinator calls such as `.take(x)` or `.skip(x)`
    /// where `x` is greater than the amount of items that an iterator will produce.
    ///
    /// ### Why is this bad?
    /// Taking or skipping more items than there are in an iterator either creates an iterator
    /// with all items from the original iterator or an iterator with no items at all.
    /// This is most likely not what the user intended to do.
    ///
    /// ### Example
    /// ```no_run
    /// for _ in [1, 2, 3].iter().take(4) {}
    /// ```
    /// Use instead:
    /// ```no_run
    /// for _ in [1, 2, 3].iter() {}
    /// ```
    #[clippy::version = "1.74.0"]
    pub ITER_OUT_OF_BOUNDS,
    suspicious,
    "calls to `.take()` or `.skip()` that are out of bounds"
}

declare_clippy_lint! {
    /// ### What it does
    /// Looks for calls to `Path::ends_with` calls where the argument looks like a file extension.
    ///
    /// By default, Clippy has a short list of known filenames that start with a dot
    /// but aren't necessarily file extensions (e.g. the `.git` folder), which are allowed by default.
    /// The `allowed-dotfiles` configuration can be used to allow additional
    /// file extensions that Clippy should not lint.
    ///
    /// ### Why is this bad?
    /// This doesn't actually compare file extensions. Rather, `ends_with` compares the given argument
    /// to the last **component** of the path and checks if it matches exactly.
    ///
    /// ### Known issues
    /// File extensions are often at most three characters long, so this only lints in those cases
    /// in an attempt to avoid false positives.
    /// Any extension names longer than that are assumed to likely be real path components and are
    /// therefore ignored.
    ///
    /// ### Example
    /// ```no_run
    /// # use std::path::Path;
    /// fn is_markdown(path: &Path) -> bool {
    ///     path.ends_with(".md")
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// # use std::path::Path;
    /// fn is_markdown(path: &Path) -> bool {
    ///     path.extension().is_some_and(|ext| ext == "md")
    /// }
    /// ```
    #[clippy::version = "1.74.0"]
    pub PATH_ENDS_WITH_EXT,
    suspicious,
    "attempting to compare file extensions using `Path::ends_with`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `as_str()` on a `String` chained with a method available on the `String` itself.
    ///
    /// ### Why is this bad?
    /// The `as_str()` conversion is pointless and can be removed for simplicity and cleanliness.
    ///
    /// ### Example
    /// ```no_run
    /// let owned_string = "This is a string".to_owned();
    /// owned_string.as_str().as_bytes()
    /// # ;
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let owned_string = "This is a string".to_owned();
    /// owned_string.as_bytes()
    /// # ;
    /// ```
    #[clippy::version = "1.74.0"]
    pub REDUNDANT_AS_STR,
    complexity,
    "`as_str` used to call a method on `str` that is also available on `String`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `waker.clone().wake()`
    ///
    /// ### Why is this bad?
    /// Cloning the waker is not necessary, `wake_by_ref()` enables the same operation
    /// without extra cloning/dropping.
    ///
    /// ### Example
    /// ```rust,ignore
    /// waker.clone().wake();
    /// ```
    /// Should be written
    /// ```rust,ignore
    /// waker.wake_by_ref();
    /// ```
    #[clippy::version = "1.75.0"]
    pub WAKER_CLONE_WAKE,
    perf,
    "cloning a `Waker` only to wake it"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to `TryInto::try_into` and `TryFrom::try_from` when their infallible counterparts
    /// could be used.
    ///
    /// ### Why is this bad?
    /// In those cases, the `TryInto` and `TryFrom` trait implementation is a blanket impl that forwards
    /// to `Into` or `From`, which always succeeds.
    /// The returned `Result<_, Infallible>` requires error handling to get the contained value
    /// even though the conversion can never fail.
    ///
    /// ### Example
    /// ```rust
    /// let _: Result<i64, _> = 1i32.try_into();
    /// let _: Result<i64, _> = <_>::try_from(1i32);
    /// ```
    /// Use `from`/`into` instead:
    /// ```rust
    /// let _: i64 = 1i32.into();
    /// let _: i64 = <_>::from(1i32);
    /// ```
    #[clippy::version = "1.75.0"]
    pub UNNECESSARY_FALLIBLE_CONVERSIONS,
    style,
    "calling the `try_from` and `try_into` trait methods when `From`/`Into` is implemented"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to `Path::join` that start with a path separator (`\\` or `/`).
    ///
    /// ### Why is this bad?
    /// If the argument to `Path::join` starts with a separator, it will overwrite
    /// the original path. If this is intentional, prefer using `Path::new` instead.
    ///
    /// Note the behavior is platform dependent. A leading `\\` will be accepted
    /// on unix systems as part of the file name
    ///
    /// See [`Path::join`](https://doc.rust-lang.org/std/path/struct.Path.html#method.join)
    ///
    /// ### Example
    /// ```rust
    /// # use std::path::{Path, PathBuf};
    /// let path = Path::new("/bin");
    /// let joined_path = path.join("/sh");
    /// assert_eq!(joined_path, PathBuf::from("/sh"));
    /// ```
    ///
    /// Use instead;
    /// ```rust
    /// # use std::path::{Path, PathBuf};
    /// let path = Path::new("/bin");
    ///
    /// // If this was unintentional, remove the leading separator
    /// let joined_path = path.join("sh");
    /// assert_eq!(joined_path, PathBuf::from("/bin/sh"));
    ///
    /// // If this was intentional, create a new path instead
    /// let new = Path::new("/sh");
    /// assert_eq!(new, PathBuf::from("/sh"));
    /// ```
    #[clippy::version = "1.76.0"]
    pub JOIN_ABSOLUTE_PATHS,
    suspicious,
    "calls to `Path::join` which will overwrite the original path"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for iterators of `Result`s using `.filter(Result::is_ok).map(Result::unwrap)` that may
    /// be replaced with a `.flatten()` call.
    ///
    /// ### Why is this bad?
    /// `Result` implements `IntoIterator<Item = T>`. This means that `Result` can be flattened
    /// automatically without suspicious-looking `unwrap` calls.
    ///
    /// ### Example
    /// ```no_run
    /// let _ = std::iter::empty::<Result<i32, ()>>().filter(Result::is_ok).map(Result::unwrap);
    /// ```
    /// Use instead:
    /// ```no_run
    /// let _ = std::iter::empty::<Result<i32, ()>>().flatten();
    /// ```
    #[clippy::version = "1.77.0"]
    pub RESULT_FILTER_MAP,
    complexity,
    "filtering `Result` for `Ok` then force-unwrapping, which can be one type-safe operation"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `.filter(Option::is_some)` that may be replaced with a `.flatten()` call.
    /// This lint will require additional changes to the follow-up calls as it affects the type.
    ///
    /// ### Why is this bad?
    /// This pattern is often followed by manual unwrapping of the `Option`. The simplification
    /// results in more readable and succinct code without the need for manual unwrapping.
    ///
    /// ### Example
    /// ```no_run
    /// vec![Some(1)].into_iter().filter(Option::is_some);
    ///
    /// ```
    /// Use instead:
    /// ```no_run
    /// vec![Some(1)].into_iter().flatten();
    /// ```
    #[clippy::version = "1.77.0"]
    pub ITER_FILTER_IS_SOME,
    pedantic,
    "filtering an iterator over `Option`s for `Some` can be achieved with `flatten`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `.filter(Result::is_ok)` that may be replaced with a `.flatten()` call.
    /// This lint will require additional changes to the follow-up calls as it affects the type.
    ///
    /// ### Why is this bad?
    /// This pattern is often followed by manual unwrapping of `Result`. The simplification
    /// results in more readable and succinct code without the need for manual unwrapping.
    ///
    /// ### Example
    /// ```no_run
    /// vec![Ok::<i32, String>(1)].into_iter().filter(Result::is_ok);
    ///
    /// ```
    /// Use instead:
    /// ```no_run
    /// vec![Ok::<i32, String>(1)].into_iter().flatten();
    /// ```
    #[clippy::version = "1.77.0"]
    pub ITER_FILTER_IS_OK,
    pedantic,
    "filtering an iterator over `Result`s for `Ok` can be achieved with `flatten`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `option.map(f).unwrap_or_default()` and `result.map(f).unwrap_or_default()` where f is a function or closure that returns the `bool` type.
    ///
    /// ### Why is this bad?
    /// Readability. These can be written more concisely as `option.is_some_and(f)` and `result.is_ok_and(f)`.
    ///
    /// ### Example
    /// ```no_run
    /// # let option = Some(1);
    /// # let result: Result<usize, ()> = Ok(1);
    /// option.map(|a| a > 10).unwrap_or_default();
    /// result.map(|a| a > 10).unwrap_or_default();
    /// ```
    /// Use instead:
    /// ```no_run
    /// # let option = Some(1);
    /// # let result: Result<usize, ()> = Ok(1);
    /// option.is_some_and(|a| a > 10);
    /// result.is_ok_and(|a| a > 10);
    /// ```
    #[clippy::version = "1.77.0"]
    pub MANUAL_IS_VARIANT_AND,
    pedantic,
    "using `.map(f).unwrap_or_default()`, which is more succinctly expressed as `is_some_and(f)` or `is_ok_and(f)`"
}

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks for usages of `str.trim().split("\n")` and `str.trim().split("\r\n")`.
    ///
    /// ### Why is this bad?
    ///
    /// Hard-coding the line endings makes the code less compatible. `str.lines` should be used instead.
    ///
    /// ### Example
    /// ```no_run
    /// "some\ntext\nwith\nnewlines\n".trim().split('\n');
    /// ```
    /// Use instead:
    /// ```no_run
    /// "some\ntext\nwith\nnewlines\n".lines();
    /// ```
    ///
    /// ### Known Problems
    ///
    /// This lint cannot detect if the split is intentionally restricted to a single type of newline (`"\n"` or
    /// `"\r\n"`), for example during the parsing of a specific file format in which precisely one newline type is
    /// valid.
    #[clippy::version = "1.77.0"]
    pub STR_SPLIT_AT_NEWLINE,
    pedantic,
    "splitting a trimmed string at hard-coded newlines"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `.as_ref().cloned()` and `.as_mut().cloned()` on `Option`s
    ///
    /// ### Why is this bad?
    /// This can be written more concisely by cloning the `Option` directly.
    ///
    /// ### Example
    /// ```no_run
    /// fn foo(bar: &Option<Vec<u8>>) -> Option<Vec<u8>> {
    ///     bar.as_ref().cloned()
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// fn foo(bar: &Option<Vec<u8>>) -> Option<Vec<u8>> {
    ///     bar.clone()
    /// }
    /// ```
    #[clippy::version = "1.77.0"]
    pub OPTION_AS_REF_CLONED,
    pedantic,
    "cloning an `Option` via `as_ref().cloned()`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for unnecessary calls to `min()` or `max()` in the following cases
    /// - Either both side is constant
    /// - One side is clearly larger than the other, like i32::MIN and an i32 variable
    ///
    /// ### Why is this bad?
    ///
    /// In the aforementioned cases it is not necessary to call `min()` or `max()`
    /// to compare values, it may even cause confusion.
    ///
    /// ### Example
    /// ```no_run
    /// let _ = 0.min(7_u32);
    /// ```
    /// Use instead:
    /// ```no_run
    /// let _ = 0;
    /// ```
    #[clippy::version = "1.81.0"]
    pub UNNECESSARY_MIN_OR_MAX,
    complexity,
    "using 'min()/max()' when there is no need for it"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `.map_or_else()` "map closure" for `Result` type.
    ///
    /// ### Why is this bad?
    /// This can be written more concisely by using `unwrap_or_else()`.
    ///
    /// ### Example
    /// ```no_run
    /// # fn handle_error(_: ()) -> u32 { 0 }
    /// let x: Result<u32, ()> = Ok(0);
    /// let y = x.map_or_else(|err| handle_error(err), |n| n);
    /// ```
    /// Use instead:
    /// ```no_run
    /// # fn handle_error(_: ()) -> u32 { 0 }
    /// let x: Result<u32, ()> = Ok(0);
    /// let y = x.unwrap_or_else(|err| handle_error(err));
    /// ```
    #[clippy::version = "1.78.0"]
    pub UNNECESSARY_RESULT_MAP_OR_ELSE,
    suspicious,
    "making no use of the \"map closure\" when calling `.map_or_else(|err| handle_error(err), |n| n)`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the manual creation of C strings (a string with a `NUL` byte at the end), either
    /// through one of the `CStr` constructor functions, or more plainly by calling `.as_ptr()`
    /// on a (byte) string literal with a hardcoded `\0` byte at the end.
    ///
    /// ### Why is this bad?
    /// This can be written more concisely using `c"str"` literals and is also less error-prone,
    /// because the compiler checks for interior `NUL` bytes and the terminating `NUL` byte is inserted automatically.
    ///
    /// ### Example
    /// ```no_run
    /// # use std::ffi::CStr;
    /// # mod libc { pub unsafe fn puts(_: *const i8) {} }
    /// fn needs_cstr(_: &CStr) {}
    ///
    /// needs_cstr(CStr::from_bytes_with_nul(b"Hello\0").unwrap());
    /// unsafe { libc::puts("World\0".as_ptr().cast()) }
    /// ```
    /// Use instead:
    /// ```no_run
    /// # use std::ffi::CStr;
    /// # mod libc { pub unsafe fn puts(_: *const i8) {} }
    /// fn needs_cstr(_: &CStr) {}
    ///
    /// needs_cstr(c"Hello");
    /// unsafe { libc::puts(c"World".as_ptr()) }
    /// ```
    #[clippy::version = "1.78.0"]
    pub MANUAL_C_STR_LITERALS,
    complexity,
    r#"creating a `CStr` through functions when `c""` literals can be used"#
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks the usage of `.get().is_some()` or `.get().is_none()` on std map types.
    ///
    /// ### Why is this bad?
    /// It can be done in one call with `.contains()`/`.contains_key()`.
    ///
    /// ### Example
    /// ```no_run
    /// # use std::collections::HashSet;
    /// let s: HashSet<String> = HashSet::new();
    /// if s.get("a").is_some() {
    ///     // code
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// # use std::collections::HashSet;
    /// let s: HashSet<String> = HashSet::new();
    /// if s.contains("a") {
    ///     // code
    /// }
    /// ```
    #[clippy::version = "1.78.0"]
    pub UNNECESSARY_GET_THEN_CHECK,
    suspicious,
    "calling `.get().is_some()` or `.get().is_none()` instead of `.contains()` or `.contains_key()`"
}

declare_clippy_lint! {
    /// ### What it does
    /// It identifies calls to `.is_empty()` on constant values.
    ///
    /// ### Why is this bad?
    /// String literals and constant values are known at compile time. Checking if they
    /// are empty will always return the same value. This might not be the intention of
    /// the expression.
    ///
    /// ### Example
    /// ```no_run
    /// let value = "";
    /// if value.is_empty() {
    ///     println!("the string is empty");
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// println!("the string is empty");
    /// ```
    #[clippy::version = "1.79.0"]
    pub CONST_IS_EMPTY,
    suspicious,
    "is_empty() called on strings known at compile time"
}

declare_clippy_lint! {
    /// ### What it does
    /// Converts some constructs mapping an Enum value for equality comparison.
    ///
    /// ### Why is this bad?
    /// Calls such as `opt.map_or(false, |val| val == 5)` are needlessly long and cumbersome,
    /// and can be reduced to, for example, `opt == Some(5)` assuming `opt` implements `PartialEq`.
    /// Also, calls such as `opt.map_or(true, |val| val == 5)` can be reduced to
    /// `opt.is_none_or(|val| val == 5)`.
    /// This lint offers readability and conciseness improvements.
    ///
    /// ### Example
    /// ```no_run
    /// pub fn a(x: Option<i32>) -> (bool, bool) {
    ///     (
    ///         x.map_or(false, |n| n == 5),
    ///         x.map_or(true, |n| n > 5),
    ///     )
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// pub fn a(x: Option<i32>) -> (bool, bool) {
    ///     (
    ///         x == Some(5),
    ///         x.is_none_or(|n| n > 5),
    ///     )
    /// }
    /// ```
    #[clippy::version = "1.84.0"]
    pub UNNECESSARY_MAP_OR,
    style,
    "reduce unnecessary calls to `.map_or(bool, …)`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks if an iterator is used to check if a string is ascii.
    ///
    /// ### Why is this bad?
    /// The `str` type already implements the `is_ascii` method.
    ///
    /// ### Example
    /// ```no_run
    /// "foo".chars().all(|c| c.is_ascii());
    /// ```
    /// Use instead:
    /// ```no_run
    /// "foo".is_ascii();
    /// ```
    #[clippy::version = "1.81.0"]
    pub NEEDLESS_CHARACTER_ITERATION,
    suspicious,
    "is_ascii() called on a char iterator"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for uses of `map` which return the original item.
    ///
    /// ### Why is this bad?
    /// `inspect` is both clearer in intent and shorter.
    ///
    /// ### Example
    /// ```no_run
    /// let x = Some(0).map(|x| { println!("{x}"); x });
    /// ```
    /// Use instead:
    /// ```no_run
    /// let x = Some(0).inspect(|x| println!("{x}"));
    /// ```
    #[clippy::version = "1.81.0"]
    pub MANUAL_INSPECT,
    complexity,
    "use of `map` returning the original item"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks the usage of `.first().is_some()` or `.first().is_none()` to check if a slice is
    /// empty.
    ///
    /// ### Why is this bad?
    /// Using `.is_empty()` is shorter and better communicates the intention.
    ///
    /// ### Example
    /// ```no_run
    /// let v = vec![1, 2, 3];
    /// if v.first().is_none() {
    ///     // The vector is empty...
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// let v = vec![1, 2, 3];
    /// if v.is_empty() {
    ///     // The vector is empty...
    /// }
    /// ```
    #[clippy::version = "1.83.0"]
    pub UNNECESSARY_FIRST_THEN_CHECK,
    complexity,
    "calling `.first().is_some()` or `.first().is_none()` instead of `.is_empty()`"
}

declare_clippy_lint! {
   /// ### What it does
   /// It detects useless calls to `str::as_bytes()` before calling `len()` or `is_empty()`.
   ///
   /// ### Why is this bad?
   /// The `len()` and `is_empty()` methods are also directly available on strings, and they
   /// return identical results. In particular, `len()` on a string returns the number of
   /// bytes.
   ///
   /// ### Example
   /// ```
   /// let len = "some string".as_bytes().len();
   /// let b = "some string".as_bytes().is_empty();
   /// ```
   /// Use instead:
   /// ```
   /// let len = "some string".len();
   /// let b = "some string".is_empty();
   /// ```
   #[clippy::version = "1.84.0"]
   pub NEEDLESS_AS_BYTES,
   complexity,
   "detect useless calls to `as_bytes()`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `.map(…)`, followed by `.all(identity)` or `.any(identity)`.
    ///
    /// ### Why is this bad?
    /// The `.all(…)` or `.any(…)` methods can be called directly in place of `.map(…)`.
    ///
    /// ### Example
    /// ```
    /// # let mut v = [""];
    /// let e1 = v.iter().map(|s| s.is_empty()).all(|a| a);
    /// let e2 = v.iter().map(|s| s.is_empty()).any(std::convert::identity);
    /// ```
    /// Use instead:
    /// ```
    /// # let mut v = [""];
    /// let e1 = v.iter().all(|s| s.is_empty());
    /// let e2 = v.iter().any(|s| s.is_empty());
    /// ```
    #[clippy::version = "1.84.0"]
    pub MAP_ALL_ANY_IDENTITY,
    complexity,
    "combine `.map(_)` followed by `.all(identity)`/`.any(identity)` into a single call"
}

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks for `Iterator::map` over ranges without using the parameter which
    /// could be more clearly expressed using `std::iter::repeat(...).take(...)`
    /// or `std::iter::repeat_n`.
    ///
    /// ### Why is this bad?
    ///
    /// It expresses the intent more clearly to `take` the correct number of times
    /// from a generating function than to apply a closure to each number in a
    /// range only to discard them.
    ///
    /// ### Example
    ///
    /// ```no_run
    /// let random_numbers : Vec<_> = (0..10).map(|_| { 3 + 1 }).collect();
    /// ```
    /// Use instead:
    /// ```no_run
    /// let f : Vec<_> = std::iter::repeat( 3 + 1 ).take(10).collect();
    /// ```
    ///
    /// ### Known Issues
    ///
    /// This lint may suggest replacing a `Map<Range>` with a `Take<RepeatWith>`.
    /// The former implements some traits that the latter does not, such as
    /// `DoubleEndedIterator`.
    #[clippy::version = "1.84.0"]
    pub MAP_WITH_UNUSED_ARGUMENT_OVER_RANGES,
    restriction,
    "map of a trivial closure (not dependent on parameter) over a range"
}

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks for `Iterator::last` being called on a  `DoubleEndedIterator`, which can be replaced
    /// with `DoubleEndedIterator::next_back`.
    ///
    /// ### Why is this bad?
    ///
    /// `Iterator::last` is implemented by consuming the iterator, which is unnecessary if
    /// the iterator is a `DoubleEndedIterator`. Since Rust traits do not allow specialization,
    /// `Iterator::last` cannot be optimized for `DoubleEndedIterator`.
    ///
    /// ### Example
    /// ```no_run
    /// let last_arg = "echo hello world".split(' ').last();
    /// ```
    /// Use instead:
    /// ```no_run
    /// let last_arg = "echo hello world".split(' ').next_back();
    /// ```
    #[clippy::version = "1.86.0"]
    pub DOUBLE_ENDED_ITERATOR_LAST,
    perf,
    "using `Iterator::last` on a `DoubleEndedIterator`"
}

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks for `NonZero*::new_unchecked()` being used in a `const` context.
    ///
    /// ### Why is this bad?
    ///
    /// Using `NonZero*::new_unchecked()` is an `unsafe` function and requires an `unsafe` context. When used in a
    /// context evaluated at compilation time, `NonZero*::new().unwrap()` will provide the same result with identical
    /// runtime performances while not requiring `unsafe`.
    ///
    /// ### Example
    /// ```no_run
    /// use std::num::NonZeroUsize;
    /// const PLAYERS: NonZeroUsize = unsafe { NonZeroUsize::new_unchecked(3) };
    /// ```
    /// Use instead:
    /// ```no_run
    /// use std::num::NonZeroUsize;
    /// const PLAYERS: NonZeroUsize = NonZeroUsize::new(3).unwrap();
    /// ```
    #[clippy::version = "1.86.0"]
    pub USELESS_NONZERO_NEW_UNCHECKED,
    complexity,
    "using `NonZero::new_unchecked()` in a `const` context"
}

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks for `repeat().take()` that can be replaced with `repeat_n()`.
    ///
    /// ### Why is this bad?
    ///
    /// Using `repeat_n()` is more concise and clearer. Also, `repeat_n()` is sometimes faster than `repeat().take()` when the type of the element is non-trivial to clone because the original value can be reused for the last `.next()` call rather than always cloning.
    ///
    /// ### Example
    /// ```no_run
    /// let _ = std::iter::repeat(10).take(3);
    /// ```
    /// Use instead:
    /// ```no_run
    /// let _ = std::iter::repeat_n(10, 3);
    /// ```
    #[clippy::version = "1.86.0"]
    pub MANUAL_REPEAT_N,
    style,
    "detect `repeat().take()` that can be replaced with `repeat_n()`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for string slices immediately followed by `as_bytes`.
    ///
    /// ### Why is this bad?
    /// It involves doing an unnecessary UTF-8 alignment check which is less efficient, and can cause a panic.
    ///
    /// ### Known problems
    /// In some cases, the UTF-8 validation and potential panic from string slicing may be required for
    /// the code's correctness. If you need to ensure the slice boundaries fall on valid UTF-8 character
    /// boundaries, the original form (`s[1..5].as_bytes()`) should be preferred.
    ///
    /// ### Example
    /// ```rust
    /// let s = "Lorem ipsum";
    /// s[1..5].as_bytes();
    /// ```
    /// Use instead:
    /// ```rust
    /// let s = "Lorem ipsum";
    /// &s.as_bytes()[1..5];
    /// ```
     #[clippy::version = "1.86.0"]
     pub SLICED_STRING_AS_BYTES,
     perf,
     "slicing a string and immediately calling as_bytes is less efficient and can lead to panics"
}

declare_clippy_lint! {
    /// ### What it does
    /// Detect functions that end with `Option::and_then` or `Result::and_then`, and suggest using
    /// the `?` operator instead.
    ///
    /// ### Why is this bad?
    /// The `and_then` method is used to chain a computation that returns an `Option` or a `Result`.
    /// This can be replaced with the `?` operator, which is more concise and idiomatic.
    ///
    /// ### Example
    ///
    /// ```no_run
    /// fn test(opt: Option<i32>) -> Option<i32> {
    ///     opt.and_then(|n| {
    ///         if n > 1 {
    ///             Some(n + 1)
    ///         } else {
    ///             None
    ///        }
    ///     })
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// fn test(opt: Option<i32>) -> Option<i32> {
    ///     let n = opt?;
    ///     if n > 1 {
    ///         Some(n + 1)
    ///     } else {
    ///         None
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.86.0"]
    pub RETURN_AND_THEN,
    restriction,
    "using `Option::and_then` or `Result::and_then` to chain a computation that returns an `Option` or a `Result`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to `Read::bytes` on types which don't implement `BufRead`.
    ///
    /// ### Why is this bad?
    /// The default implementation calls `read` for each byte, which can be very inefficient for data that’s not in memory, such as `File`.
    ///
    /// ### Example
    /// ```no_run
    /// use std::io::Read;
    /// use std::fs::File;
    /// let file = File::open("./bytes.txt").unwrap();
    /// file.bytes();
    /// ```
    /// Use instead:
    /// ```no_run
    /// use std::io::{BufReader, Read};
    /// use std::fs::File;
    /// let file = BufReader::new(File::open("./bytes.txt").unwrap());
    /// file.bytes();
    /// ```
    #[clippy::version = "1.87.0"]
    pub UNBUFFERED_BYTES,
    perf,
    "calling .bytes() is very inefficient when data is not in memory"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `iter().any()` on slices when it can be replaced with `contains()` and suggests doing so.
    ///
    /// ### Why is this bad?
    /// `contains()` is more concise and idiomatic, while also being faster in some cases.
    ///
    /// ### Example
    /// ```no_run
    /// fn foo(values: &[u8]) -> bool {
    ///     values.iter().any(|&v| v == 10)
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// fn foo(values: &[u8]) -> bool {
    ///     values.contains(&10)
    /// }
    /// ```
    #[clippy::version = "1.87.0"]
    pub MANUAL_CONTAINS,
    perf,
    "unnecessary `iter().any()` on slices that can be replaced with `contains()`"
}

declare_clippy_lint! {
    /// This lint warns on calling `io::Error::new(..)` with a kind of
    /// `io::ErrorKind::Other`.
    ///
    /// ### Why is this bad?
    /// Since Rust 1.74, there's the `io::Error::other(_)` shortcut.
    ///
    /// ### Example
    /// ```no_run
    /// use std::io;
    /// let _ = io::Error::new(io::ErrorKind::Other, "bad".to_string());
    /// ```
    /// Use instead:
    /// ```no_run
    /// let _ = std::io::Error::other("bad".to_string());
    /// ```
    #[clippy::version = "1.87.0"]
    pub IO_OTHER_ERROR,
    style,
    "calling `std::io::Error::new(std::io::ErrorKind::Other, _)`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `std::mem::swap` with temporary values.
    ///
    /// ### Why is this bad?
    /// Storing a new value in place of a temporary value which will
    /// be dropped right after the `swap` is an inefficient way of performing
    /// an assignment. The same result can be achieved by using a regular
    /// assignment.
    ///
    /// ### Examples
    /// ```no_run
    /// fn replace_string(s: &mut String) {
    ///     std::mem::swap(s, &mut String::from("replaced"));
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// fn replace_string(s: &mut String) {
    ///     *s = String::from("replaced");
    /// }
    /// ```
    ///
    /// Also, swapping two temporary values has no effect, as they will
    /// both be dropped right after swapping them. This is likely an indication
    /// of a bug. For example, the following code swaps the references to
    /// the last element of the vectors, instead of swapping the elements
    /// themselves:
    ///
    /// ```no_run
    /// fn bug(v1: &mut [i32], v2: &mut [i32]) {
    ///     // Incorrect: swapping temporary references (`&mut &mut` passed to swap)
    ///     std::mem::swap(&mut v1.last_mut().unwrap(), &mut v2.last_mut().unwrap());
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// fn correct(v1: &mut [i32], v2: &mut [i32]) {
    ///     std::mem::swap(v1.last_mut().unwrap(), v2.last_mut().unwrap());
    /// }
    /// ```
    #[clippy::version = "1.88.0"]
    pub SWAP_WITH_TEMPORARY,
    complexity,
    "detect swap with a temporary value"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for IP addresses that could be replaced with predefined constants such as
    /// `Ipv4Addr::new(127, 0, 0, 1)` instead of using the appropriate constants.
    ///
    /// ### Why is this bad?
    /// Using specific IP addresses like `127.0.0.1` or `::1` is less clear and less maintainable than using the
    /// predefined constants `Ipv4Addr::LOCALHOST` or `Ipv6Addr::LOCALHOST`. These constants improve code
    /// readability, make the intent explicit, and are less error-prone.
    ///
    /// ### Example
    /// ```no_run
    /// use std::net::{Ipv4Addr, Ipv6Addr};
    ///
    /// // IPv4 loopback
    /// let addr_v4 = Ipv4Addr::new(127, 0, 0, 1);
    ///
    /// // IPv6 loopback
    /// let addr_v6 = Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1);
    /// ```
    /// Use instead:
    /// ```no_run
    /// use std::net::{Ipv4Addr, Ipv6Addr};
    ///
    /// // IPv4 loopback
    /// let addr_v4 = Ipv4Addr::LOCALHOST;
    ///
    /// // IPv6 loopback
    /// let addr_v6 = Ipv6Addr::LOCALHOST;
    /// ```
    #[clippy::version = "1.89.0"]
    pub IP_CONSTANT,
    pedantic,
    "hardcoded localhost IP address"
}

#[expect(clippy::struct_excessive_bools)]
pub struct Methods {
    avoid_breaking_exported_api: bool,
    msrv: Msrv,
    allow_expect_in_tests: bool,
    allow_unwrap_in_tests: bool,
    allow_expect_in_consts: bool,
    allow_unwrap_in_consts: bool,
    allowed_dotfiles: FxHashSet<&'static str>,
    format_args: FormatArgsStorage,
}

impl Methods {
    pub fn new(conf: &'static Conf, format_args: FormatArgsStorage) -> Self {
        let mut allowed_dotfiles: FxHashSet<_> = conf.allowed_dotfiles.iter().map(|s| &**s).collect();
        allowed_dotfiles.extend(DEFAULT_ALLOWED_DOTFILES);

        Self {
            avoid_breaking_exported_api: conf.avoid_breaking_exported_api,
            msrv: conf.msrv,
            allow_expect_in_tests: conf.allow_expect_in_tests,
            allow_unwrap_in_tests: conf.allow_unwrap_in_tests,
            allow_expect_in_consts: conf.allow_expect_in_consts,
            allow_unwrap_in_consts: conf.allow_unwrap_in_consts,
            allowed_dotfiles,
            format_args,
        }
    }
}

impl_lint_pass!(Methods => [
    UNWRAP_USED,
    EXPECT_USED,
    SHOULD_IMPLEMENT_TRAIT,
    WRONG_SELF_CONVENTION,
    OK_EXPECT,
    UNWRAP_OR_DEFAULT,
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
    COLLAPSIBLE_STR_REPLACE,
    CONST_IS_EMPTY,
    ITER_OVEREAGER_CLONED,
    CLONED_INSTEAD_OF_COPIED,
    FLAT_MAP_OPTION,
    INEFFICIENT_TO_STRING,
    NEW_RET_NO_SELF,
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
    TYPE_ID_ON_BOX,
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
    SUSPICIOUS_OPEN_OPTIONS,
    PATH_BUF_PUSH_OVERWRITE,
    RANGE_ZIP_WITH_LEN,
    REPEAT_ONCE,
    STABLE_SORT_PRIMITIVE,
    UNIT_HASH,
    READ_LINE_WITHOUT_TRIM,
    UNNECESSARY_SORT_BY,
    VEC_RESIZE_TO_ZERO,
    VERBOSE_FILE_READS,
    ITER_KV_MAP,
    SEEK_FROM_CURRENT,
    SEEK_TO_START_INSTEAD_OF_REWIND,
    NEEDLESS_COLLECT,
    SUSPICIOUS_COMMAND_ARG_SPACE,
    CLEAR_WITH_DRAIN,
    MANUAL_NEXT_BACK,
    UNNECESSARY_LITERAL_UNWRAP,
    DRAIN_COLLECT,
    MANUAL_TRY_FOLD,
    FORMAT_COLLECT,
    STRING_LIT_CHARS_ANY,
    ITER_SKIP_ZERO,
    FILTER_MAP_BOOL_THEN,
    READONLY_WRITE_LOCK,
    ITER_OUT_OF_BOUNDS,
    PATH_ENDS_WITH_EXT,
    REDUNDANT_AS_STR,
    WAKER_CLONE_WAKE,
    UNNECESSARY_FALLIBLE_CONVERSIONS,
    JOIN_ABSOLUTE_PATHS,
    RESULT_FILTER_MAP,
    ITER_FILTER_IS_SOME,
    ITER_FILTER_IS_OK,
    MANUAL_IS_VARIANT_AND,
    STR_SPLIT_AT_NEWLINE,
    OPTION_AS_REF_CLONED,
    UNNECESSARY_RESULT_MAP_OR_ELSE,
    MANUAL_C_STR_LITERALS,
    UNNECESSARY_GET_THEN_CHECK,
    UNNECESSARY_FIRST_THEN_CHECK,
    NEEDLESS_CHARACTER_ITERATION,
    MANUAL_INSPECT,
    UNNECESSARY_MIN_OR_MAX,
    NEEDLESS_AS_BYTES,
    MAP_ALL_ANY_IDENTITY,
    MAP_WITH_UNUSED_ARGUMENT_OVER_RANGES,
    UNNECESSARY_MAP_OR,
    DOUBLE_ENDED_ITERATOR_LAST,
    USELESS_NONZERO_NEW_UNCHECKED,
    MANUAL_REPEAT_N,
    SLICED_STRING_AS_BYTES,
    RETURN_AND_THEN,
    UNBUFFERED_BYTES,
    MANUAL_CONTAINS,
    IO_OTHER_ERROR,
    SWAP_WITH_TEMPORARY,
    IP_CONSTANT,
]);

/// Extracts a method call name, args, and `Span` of the method name.
/// This ensures that neither the receiver nor any of the arguments
/// come from expansion.
pub fn method_call<'tcx>(recv: &'tcx Expr<'tcx>) -> Option<(Symbol, &'tcx Expr<'tcx>, &'tcx [Expr<'tcx>], Span, Span)> {
    if let ExprKind::MethodCall(path, receiver, args, call_span) = recv.kind
        && !args.iter().any(|e| e.span.from_expansion())
        && !receiver.span.from_expansion()
    {
        Some((path.ident.name, receiver, args, path.ident.span, call_span))
    } else {
        None
    }
}

impl<'tcx> LateLintPass<'tcx> for Methods {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if expr.span.from_expansion() {
            return;
        }

        self.check_methods(cx, expr);

        match expr.kind {
            ExprKind::Call(func, args) => {
                from_iter_instead_of_collect::check(cx, expr, args, func);
                unnecessary_fallible_conversions::check_function(cx, expr, func);
                manual_c_str_literals::check(cx, expr, func, args, self.msrv);
                useless_nonzero_new_unchecked::check(cx, expr, func, args, self.msrv);
                io_other_error::check(cx, expr, func, args, self.msrv);
                swap_with_temporary::check(cx, expr, func, args);
                ip_constant::check(cx, expr, func, args);
            },
            ExprKind::MethodCall(method_call, receiver, args, _) => {
                let method_span = method_call.ident.span;
                or_fun_call::check(cx, expr, method_span, method_call.ident.name, receiver, args);
                expect_fun_call::check(
                    cx,
                    &self.format_args,
                    expr,
                    method_span,
                    method_call.ident.name,
                    receiver,
                    args,
                );
                clone_on_copy::check(cx, expr, method_call.ident.name, receiver, args);
                clone_on_ref_ptr::check(cx, expr, method_call.ident.name, receiver, args);
                inefficient_to_string::check(cx, expr, method_call.ident.name, receiver, args);
                single_char_add_str::check(cx, expr, receiver, args);
                into_iter_on_ref::check(cx, expr, method_span, method_call.ident.name, receiver);
                unnecessary_to_owned::check(cx, expr, method_call.ident.name, receiver, args, self.msrv);
            },
            ExprKind::Binary(op, lhs, rhs) if op.node == hir::BinOpKind::Eq || op.node == hir::BinOpKind::Ne => {
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
        if impl_item.span.in_external_macro(cx.sess().source_map()) {
            return;
        }
        let name = impl_item.ident.name;
        let parent = cx.tcx.hir_get_parent_item(impl_item.hir_id()).def_id;
        let item = cx.tcx.hir_expect_item(parent);
        let self_ty = cx.tcx.type_of(item.owner_id).instantiate_identity();

        let implements_trait = matches!(item.kind, hir::ItemKind::Impl(hir::Impl { of_trait: Some(_), .. }));
        if let hir::ImplItemKind::Fn(ref sig, id) = impl_item.kind {
            let method_sig = cx.tcx.fn_sig(impl_item.owner_id).instantiate_identity();
            let method_sig = cx.tcx.instantiate_bound_regions_with_erased(method_sig);
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
                        && first_arg_ty_opt.is_none_or(|first_arg_ty| method_config
                            .self_kind.matches(cx, self_ty, first_arg_ty)
                            )
                        && fn_header_equals(method_config.fn_header, sig.header)
                        && method_config.lifetime_param_cond(impl_item)
                    {
                        span_lint_and_help(
                            cx,
                            SHOULD_IMPLEMENT_TRAIT,
                            impl_item.span,
                            format!(
                                "method `{}` can be confused for the standard trait method `{}::{}`",
                                method_config.method_name, method_config.trait_name, method_config.method_name
                            ),
                            None,
                            format!(
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
                && let Some(first_arg) = iter_input_pats(sig.decl, cx.tcx.hir_body(id)).next()
                && let Some(first_arg_ty) = first_arg_ty_opt
            {
                wrong_self_convention::check(
                    cx,
                    name,
                    self_ty,
                    first_arg_ty,
                    first_arg.pat.span,
                    implements_trait,
                    false,
                );
            }
        }

        // if this impl block implements a trait, lint in trait definition instead
        if implements_trait {
            return;
        }

        if let hir::ImplItemKind::Fn(_, _) = impl_item.kind {
            let ret_ty = return_ty(cx, impl_item.owner_id);

            if contains_ty_adt_constructor_opaque(cx, ret_ty, self_ty) {
                return;
            }

            if name == sym::new && ret_ty != self_ty {
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
        if item.span.in_external_macro(cx.tcx.sess.source_map()) {
            return;
        }

        if let TraitItemKind::Fn(ref sig, _) = item.kind
            && sig.decl.implicit_self.has_implicit_self()
            && let Some(first_arg_hir_ty) = sig.decl.inputs.first()
            && let Some(&first_arg_ty) = cx
                .tcx
                .fn_sig(item.owner_id)
                .instantiate_identity()
                .inputs()
                .skip_binder()
                .first()
        {
            let self_ty = TraitRef::identity(cx.tcx, item.owner_id.to_def_id()).self_ty();
            wrong_self_convention::check(
                cx,
                item.ident.name,
                self_ty,
                first_arg_ty,
                first_arg_hir_ty.span,
                false,
                true,
            );
        }

        if item.ident.name == sym::new
            && let TraitItemKind::Fn(_, _) = item.kind
            && let ret_ty = return_ty(cx, item.owner_id)
            && let self_ty = TraitRef::identity(cx.tcx, item.owner_id.to_def_id()).self_ty()
            && !ret_ty.contains(self_ty)
        {
            span_lint(
                cx,
                NEW_RET_NO_SELF,
                item.span,
                "methods called `new` usually return `Self`",
            );
        }
    }
}

impl Methods {
    #[allow(clippy::too_many_lines)]
    fn check_methods<'tcx>(&self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        // Handle method calls whose receiver and arguments may not come from expansion
        if let Some((name, recv, args, span, call_span)) = method_call(expr) {
            match (name, args) {
                (
                    sym::add | sym::offset | sym::sub | sym::wrapping_offset | sym::wrapping_add | sym::wrapping_sub,
                    [_arg],
                ) => {
                    zst_offset::check(cx, expr, recv);
                },
                (sym::all, [arg]) => {
                    unused_enumerate_index::check(cx, expr, recv, arg);
                    needless_character_iteration::check(cx, expr, recv, arg, true);
                    match method_call(recv) {
                        Some((sym::cloned, recv2, [], _, _)) => {
                            iter_overeager_cloned::check(
                                cx,
                                expr,
                                recv,
                                recv2,
                                iter_overeager_cloned::Op::NeedlessMove(arg),
                                false,
                            );
                        },
                        Some((sym::map, _, [map_arg], _, map_call_span)) => {
                            map_all_any_identity::check(cx, expr, recv, map_call_span, map_arg, call_span, arg, "all");
                        },
                        _ => {},
                    }
                },
                (sym::and_then, [arg]) => {
                    let biom_option_linted = bind_instead_of_map::check_and_then_some(cx, expr, recv, arg);
                    let biom_result_linted = bind_instead_of_map::check_and_then_ok(cx, expr, recv, arg);
                    if !biom_option_linted && !biom_result_linted {
                        let ule_and_linted = unnecessary_lazy_eval::check(cx, expr, recv, arg, "and");
                        if !ule_and_linted {
                            return_and_then::check(cx, expr, recv, arg);
                        }
                    }
                },
                (sym::any, [arg]) => {
                    unused_enumerate_index::check(cx, expr, recv, arg);
                    needless_character_iteration::check(cx, expr, recv, arg, false);
                    match method_call(recv) {
                        Some((sym::cloned, recv2, [], _, _)) => iter_overeager_cloned::check(
                            cx,
                            expr,
                            recv,
                            recv2,
                            iter_overeager_cloned::Op::NeedlessMove(arg),
                            false,
                        ),
                        Some((sym::chars, recv, _, _, _))
                            if let ExprKind::Closure(arg) = arg.kind
                                && let body = cx.tcx.hir_body(arg.body)
                                && let [param] = body.params =>
                        {
                            string_lit_chars_any::check(cx, expr, recv, param, peel_blocks(body.value), self.msrv);
                        },
                        Some((sym::map, _, [map_arg], _, map_call_span)) => {
                            map_all_any_identity::check(cx, expr, recv, map_call_span, map_arg, call_span, arg, "any");
                        },
                        Some((sym::iter, iter_recv, ..)) => {
                            manual_contains::check(cx, expr, iter_recv, arg);
                        },
                        _ => {},
                    }
                },
                (sym::arg, [arg]) => {
                    suspicious_command_arg_space::check(cx, recv, arg, span);
                },
                (sym::as_deref | sym::as_deref_mut, []) => {
                    needless_option_as_deref::check(cx, expr, recv, name);
                },
                (sym::as_bytes, []) => {
                    if let Some((sym::as_str, recv, [], as_str_span, _)) = method_call(recv) {
                        redundant_as_str::check(cx, expr, recv, as_str_span, span);
                    }
                    sliced_string_as_bytes::check(cx, expr, recv);
                },
                (sym::as_mut | sym::as_ref, []) => useless_asref::check(cx, expr, name, recv),
                (sym::as_ptr, []) => manual_c_str_literals::check_as_ptr(cx, expr, recv, self.msrv),
                (sym::assume_init, []) => uninit_assumed_init::check(cx, expr, recv),
                (sym::bytes, []) => unbuffered_bytes::check(cx, expr, recv),
                (sym::cloned, []) => {
                    cloned_instead_of_copied::check(cx, expr, recv, span, self.msrv);
                    option_as_ref_cloned::check(cx, recv, span);
                },
                (sym::collect, []) if is_trait_method(cx, expr, sym::Iterator) => {
                    needless_collect::check(cx, span, expr, recv, call_span);
                    match method_call(recv) {
                        Some((name @ (sym::cloned | sym::copied), recv2, [], _, _)) => {
                            iter_cloned_collect::check(cx, name, expr, recv2);
                        },
                        Some((sym::map, m_recv, [m_arg], m_ident_span, _)) => {
                            map_collect_result_unit::check(cx, expr, m_recv, m_arg);
                            format_collect::check(cx, expr, m_arg, m_ident_span);
                        },
                        Some((sym::take, take_self_arg, [take_arg], _, _)) => {
                            if self.msrv.meets(cx, msrvs::STR_REPEAT) {
                                manual_str_repeat::check(cx, expr, recv, take_self_arg, take_arg);
                            }
                        },
                        Some((sym::drain, recv, args, ..)) => {
                            drain_collect::check(cx, args, expr, recv);
                        },
                        _ => {},
                    }
                },
                (sym::count, []) if is_trait_method(cx, expr, sym::Iterator) => match method_call(recv) {
                    Some((sym::cloned, recv2, [], _, _)) => {
                        iter_overeager_cloned::check(cx, expr, recv, recv2, iter_overeager_cloned::Op::RmCloned, false);
                    },
                    Some((name2 @ (sym::into_iter | sym::iter | sym::iter_mut), recv2, [], _, _)) => {
                        iter_count::check(cx, expr, recv2, name2);
                    },
                    Some((sym::map, _, [arg], _, _)) => suspicious_map::check(cx, expr, recv, arg),
                    Some((sym::filter, recv2, [arg], _, _)) => bytecount::check(cx, expr, recv2, arg),
                    Some((sym::bytes, recv2, [], _, _)) => bytes_count_to_len::check(cx, expr, recv, recv2),
                    _ => {},
                },
                (sym::min | sym::max, [arg]) => {
                    unnecessary_min_or_max::check(cx, expr, name, recv, arg);
                },
                (sym::drain, ..) => {
                    if let Node::Stmt(Stmt { hir_id: _, kind, .. }) = cx.tcx.parent_hir_node(expr.hir_id)
                        && matches!(kind, StmtKind::Semi(_))
                        && args.len() <= 1
                    {
                        clear_with_drain::check(cx, expr, recv, span, args.first());
                    } else if let [arg] = args {
                        iter_with_drain::check(cx, expr, recv, span, arg);
                    }
                },
                (sym::ends_with, [arg]) => {
                    if let ExprKind::MethodCall(.., span) = expr.kind {
                        case_sensitive_file_extension_comparisons::check(cx, expr, span, recv, arg, self.msrv);
                    }
                    path_ends_with_ext::check(cx, recv, arg, expr, self.msrv, &self.allowed_dotfiles);
                },
                (sym::expect, [_]) => {
                    match method_call(recv) {
                        Some((sym::ok, recv, [], _, _)) => ok_expect::check(cx, expr, recv),
                        Some((sym::err, recv, [], err_span, _)) => {
                            err_expect::check(cx, expr, recv, span, err_span, self.msrv);
                        },
                        _ => {},
                    }
                    unnecessary_literal_unwrap::check(cx, expr, recv, name, args);
                },
                (sym::expect_err, [_]) | (sym::unwrap_err | sym::unwrap_unchecked | sym::unwrap_err_unchecked, []) => {
                    unnecessary_literal_unwrap::check(cx, expr, recv, name, args);
                },
                (sym::extend, [arg]) => {
                    string_extend_chars::check(cx, expr, recv, arg);
                    extend_with_drain::check(cx, expr, recv, arg);
                },
                (sym::filter, [arg]) => {
                    if let Some((sym::cloned, recv2, [], _span2, _)) = method_call(recv) {
                        // if `arg` has side-effect, the semantic will change
                        iter_overeager_cloned::check(
                            cx,
                            expr,
                            recv,
                            recv2,
                            iter_overeager_cloned::Op::FixClosure(name, arg),
                            false,
                        );
                    }
                    if self.msrv.meets(cx, msrvs::ITER_FLATTEN) {
                        // use the sourcemap to get the span of the closure
                        iter_filter::check(cx, expr, arg, span);
                    }
                },
                (sym::find, [arg]) => {
                    if let Some((sym::cloned, recv2, [], _span2, _)) = method_call(recv) {
                        // if `arg` has side-effect, the semantic will change
                        iter_overeager_cloned::check(
                            cx,
                            expr,
                            recv,
                            recv2,
                            iter_overeager_cloned::Op::FixClosure(name, arg),
                            false,
                        );
                    }
                },
                (sym::filter_map, [arg]) => {
                    unused_enumerate_index::check(cx, expr, recv, arg);
                    unnecessary_filter_map::check(cx, expr, arg, name);
                    filter_map_bool_then::check(cx, expr, arg, call_span);
                    filter_map_identity::check(cx, expr, arg, span);
                },
                (sym::find_map, [arg]) => {
                    unused_enumerate_index::check(cx, expr, recv, arg);
                    unnecessary_filter_map::check(cx, expr, arg, name);
                },
                (sym::flat_map, [arg]) => {
                    unused_enumerate_index::check(cx, expr, recv, arg);
                    flat_map_identity::check(cx, expr, arg, span);
                    flat_map_option::check(cx, expr, arg, span);
                },
                (sym::flatten, []) => match method_call(recv) {
                    Some((sym::map, recv, [map_arg], map_span, _)) => {
                        map_flatten::check(cx, expr, recv, map_arg, map_span);
                    },
                    Some((sym::cloned, recv2, [], _, _)) => iter_overeager_cloned::check(
                        cx,
                        expr,
                        recv,
                        recv2,
                        iter_overeager_cloned::Op::LaterCloned,
                        true,
                    ),
                    _ => {},
                },
                (sym::fold, [init, acc]) => {
                    manual_try_fold::check(cx, expr, init, acc, call_span, self.msrv);
                    unnecessary_fold::check(cx, expr, init, acc, span);
                },
                (sym::for_each, [arg]) => {
                    unused_enumerate_index::check(cx, expr, recv, arg);
                    match method_call(recv) {
                        Some((sym::inspect, _, [_], span2, _)) => inspect_for_each::check(cx, expr, span2),
                        Some((sym::cloned, recv2, [], _, _)) => iter_overeager_cloned::check(
                            cx,
                            expr,
                            recv,
                            recv2,
                            iter_overeager_cloned::Op::NeedlessMove(arg),
                            false,
                        ),
                        _ => {},
                    }
                },
                (sym::get, [arg]) => {
                    get_first::check(cx, expr, recv, arg);
                    get_last_with_len::check(cx, expr, recv, arg);
                },
                (sym::get_or_insert_with, [arg]) => {
                    unnecessary_lazy_eval::check(cx, expr, recv, arg, "get_or_insert");
                },
                (sym::hash, [arg]) => {
                    unit_hash::check(cx, expr, recv, arg);
                },
                (sym::is_empty, []) => {
                    match method_call(recv) {
                        Some((prev_method @ (sym::as_bytes | sym::bytes), prev_recv, [], _, _)) => {
                            needless_as_bytes::check(cx, prev_method, name, prev_recv, expr.span);
                        },
                        Some((sym::as_str, recv, [], as_str_span, _)) => {
                            redundant_as_str::check(cx, expr, recv, as_str_span, span);
                        },
                        _ => {},
                    }
                    is_empty::check(cx, expr, recv);
                },
                (sym::is_file, []) => filetype_is_file::check(cx, expr, recv),
                (sym::is_digit, [radix]) => is_digit_ascii_radix::check(cx, expr, recv, radix, self.msrv),
                (sym::is_none, []) => check_is_some_is_none(cx, expr, recv, call_span, false),
                (sym::is_some, []) => check_is_some_is_none(cx, expr, recv, call_span, true),
                (sym::iter | sym::iter_mut | sym::into_iter, []) => {
                    iter_on_single_or_empty_collections::check(cx, expr, name, recv);
                },
                (sym::join, [join_arg]) => {
                    if let Some((sym::collect, _, _, span, _)) = method_call(recv) {
                        unnecessary_join::check(cx, expr, recv, join_arg, span);
                    } else {
                        join_absolute_paths::check(cx, recv, join_arg, expr.span);
                    }
                },
                (sym::last, []) => {
                    if let Some((sym::cloned, recv2, [], _span2, _)) = method_call(recv) {
                        iter_overeager_cloned::check(
                            cx,
                            expr,
                            recv,
                            recv2,
                            iter_overeager_cloned::Op::LaterCloned,
                            false,
                        );
                    }
                    double_ended_iterator_last::check(cx, expr, recv, call_span);
                },
                (sym::len, []) => {
                    if let Some((prev_method @ (sym::as_bytes | sym::bytes), prev_recv, [], _, _)) = method_call(recv) {
                        needless_as_bytes::check(cx, prev_method, sym::len, prev_recv, expr.span);
                    }
                },
                (sym::lock, []) => {
                    mut_mutex_lock::check(cx, expr, recv, span);
                },
                (name @ (sym::map | sym::map_err), [m_arg]) => {
                    if name == sym::map {
                        unused_enumerate_index::check(cx, expr, recv, m_arg);
                        map_clone::check(cx, expr, recv, m_arg, self.msrv);
                        map_with_unused_argument_over_ranges::check(cx, expr, recv, m_arg, self.msrv, span);
                        manual_is_variant_and::check_map(cx, expr);
                        match method_call(recv) {
                            Some((map_name @ (sym::iter | sym::into_iter), recv2, _, _, _)) => {
                                iter_kv_map::check(cx, map_name, expr, recv2, m_arg, self.msrv);
                            },
                            Some((sym::cloned, recv2, [], _, _)) => iter_overeager_cloned::check(
                                cx,
                                expr,
                                recv,
                                recv2,
                                iter_overeager_cloned::Op::NeedlessMove(m_arg),
                                false,
                            ),
                            _ => {},
                        }
                    } else {
                        map_err_ignore::check(cx, expr, m_arg);
                    }
                    if let Some((name, recv2, args, span2, _)) = method_call(recv) {
                        match (name, args) {
                            (sym::as_mut, []) => option_as_ref_deref::check(cx, expr, recv2, m_arg, true, self.msrv),
                            (sym::as_ref, []) => option_as_ref_deref::check(cx, expr, recv2, m_arg, false, self.msrv),
                            (sym::filter, [f_arg]) => {
                                filter_map::check(cx, expr, recv2, f_arg, span2, recv, m_arg, span, false);
                            },
                            (sym::find, [f_arg]) => {
                                filter_map::check(cx, expr, recv2, f_arg, span2, recv, m_arg, span, true);
                            },
                            _ => {},
                        }
                    }
                    map_identity::check(cx, expr, recv, m_arg, name, span);
                    manual_inspect::check(cx, expr, m_arg, name, span, self.msrv);
                    crate::useless_conversion::check_function_application(cx, expr, recv, m_arg);
                },
                (sym::map_break | sym::map_continue, [m_arg]) => {
                    crate::useless_conversion::check_function_application(cx, expr, recv, m_arg);
                },
                (sym::map_or, [def, map]) => {
                    option_map_or_none::check(cx, expr, recv, def, map);
                    manual_ok_or::check(cx, expr, recv, def, map);
                    unnecessary_map_or::check(cx, expr, recv, def, map, span, self.msrv);
                },
                (sym::map_or_else, [def, map]) => {
                    result_map_or_else_none::check(cx, expr, recv, def, map);
                    unnecessary_result_map_or_else::check(cx, expr, recv, def, map);
                },
                (sym::next, []) => {
                    if let Some((name2, recv2, args2, _, _)) = method_call(recv) {
                        match (name2, args2) {
                            (sym::cloned, []) => iter_overeager_cloned::check(
                                cx,
                                expr,
                                recv,
                                recv2,
                                iter_overeager_cloned::Op::LaterCloned,
                                false,
                            ),
                            (sym::filter, [arg]) => filter_next::check(cx, expr, recv2, arg),
                            (sym::filter_map, [arg]) => filter_map_next::check(cx, expr, recv2, arg, self.msrv),
                            (sym::iter, []) => iter_next_slice::check(cx, expr, recv2),
                            (sym::skip, [arg]) => iter_skip_next::check(cx, expr, recv2, arg),
                            (sym::skip_while, [_]) => skip_while_next::check(cx, expr),
                            (sym::rev, []) => manual_next_back::check(cx, expr, recv, recv2),
                            _ => {},
                        }
                    }
                },
                (sym::nth, [n_arg]) => match method_call(recv) {
                    Some((sym::bytes, recv2, [], _, _)) => bytes_nth::check(cx, expr, recv2, n_arg),
                    Some((sym::cloned, recv2, [], _, _)) => iter_overeager_cloned::check(
                        cx,
                        expr,
                        recv,
                        recv2,
                        iter_overeager_cloned::Op::LaterCloned,
                        false,
                    ),
                    Some((iter_method @ (sym::iter | sym::iter_mut), iter_recv, [], iter_span, _)) => {
                        if !iter_nth::check(cx, expr, iter_recv, iter_method, iter_span, span) {
                            iter_nth_zero::check(cx, expr, recv, n_arg);
                        }
                    },
                    _ => iter_nth_zero::check(cx, expr, recv, n_arg),
                },
                (sym::ok_or_else, [arg]) => {
                    unnecessary_lazy_eval::check(cx, expr, recv, arg, "ok_or");
                },
                (sym::open, [_]) => {
                    open_options::check(cx, expr, recv);
                },
                (sym::or_else, [arg]) => {
                    if !bind_instead_of_map::check_or_else_err(cx, expr, recv, arg) {
                        unnecessary_lazy_eval::check(cx, expr, recv, arg, "or");
                    }
                },
                (sym::push, [arg]) => {
                    path_buf_push_overwrite::check(cx, expr, arg);
                },
                (sym::read_to_end, [_]) => {
                    verbose_file_reads::check(cx, expr, recv, verbose_file_reads::READ_TO_END_MSG);
                },
                (sym::read_to_string, [_]) => {
                    verbose_file_reads::check(cx, expr, recv, verbose_file_reads::READ_TO_STRING_MSG);
                },
                (sym::read_line, [arg]) => {
                    read_line_without_trim::check(cx, expr, recv, arg);
                },
                (sym::repeat, [arg]) => {
                    repeat_once::check(cx, expr, recv, arg);
                },
                (name @ (sym::replace | sym::replacen), [arg1, arg2] | [arg1, arg2, _]) => {
                    no_effect_replace::check(cx, expr, arg1, arg2);

                    // Check for repeated `str::replace` calls to perform `collapsible_str_replace` lint
                    if self.msrv.meets(cx, msrvs::PATTERN_TRAIT_CHAR_ARRAY)
                        && name == sym::replace
                        && let Some((sym::replace, ..)) = method_call(recv)
                    {
                        collapsible_str_replace::check(cx, expr, arg1, arg2);
                    }
                },
                (sym::resize, [count_arg, default_arg]) => {
                    vec_resize_to_zero::check(cx, expr, count_arg, default_arg, span);
                },
                (sym::seek, [arg]) => {
                    if self.msrv.meets(cx, msrvs::SEEK_FROM_CURRENT) {
                        seek_from_current::check(cx, expr, recv, arg);
                    }
                    if self.msrv.meets(cx, msrvs::SEEK_REWIND) {
                        seek_to_start_instead_of_rewind::check(cx, expr, recv, arg, span);
                    }
                },
                (sym::skip, [arg]) => {
                    iter_skip_zero::check(cx, expr, arg);
                    iter_out_of_bounds::check_skip(cx, expr, recv, arg);

                    if let Some((sym::cloned, recv2, [], _span2, _)) = method_call(recv) {
                        iter_overeager_cloned::check(
                            cx,
                            expr,
                            recv,
                            recv2,
                            iter_overeager_cloned::Op::LaterCloned,
                            false,
                        );
                    }
                },
                (sym::sort, []) => {
                    stable_sort_primitive::check(cx, expr, recv);
                },
                (sym::sort_by, [arg]) => {
                    unnecessary_sort_by::check(cx, expr, recv, arg, false);
                },
                (sym::sort_unstable_by, [arg]) => {
                    unnecessary_sort_by::check(cx, expr, recv, arg, true);
                },
                (sym::split, [arg]) => {
                    str_split::check(cx, expr, recv, arg);
                },
                (sym::splitn | sym::rsplitn, [count_arg, pat_arg]) => {
                    if let Some(Constant::Int(count)) = ConstEvalCtxt::new(cx).eval(count_arg) {
                        suspicious_splitn::check(cx, name, expr, recv, count);
                        str_splitn::check(cx, name, expr, recv, pat_arg, count, self.msrv);
                    }
                },
                (sym::splitn_mut | sym::rsplitn_mut, [count_arg, _]) => {
                    if let Some(Constant::Int(count)) = ConstEvalCtxt::new(cx).eval(count_arg) {
                        suspicious_splitn::check(cx, name, expr, recv, count);
                    }
                },
                (sym::step_by, [arg]) => iterator_step_by_zero::check(cx, expr, arg),
                (sym::take, [arg]) => {
                    iter_out_of_bounds::check_take(cx, expr, recv, arg);
                    manual_repeat_n::check(cx, expr, recv, arg, self.msrv);
                    if let Some((sym::cloned, recv2, [], _span2, _)) = method_call(recv) {
                        iter_overeager_cloned::check(
                            cx,
                            expr,
                            recv,
                            recv2,
                            iter_overeager_cloned::Op::LaterCloned,
                            false,
                        );
                    }
                },
                (sym::take, []) => needless_option_take::check(cx, expr, recv),
                (sym::then, [arg]) => {
                    if !self.msrv.meets(cx, msrvs::BOOL_THEN_SOME) {
                        return;
                    }
                    unnecessary_lazy_eval::check(cx, expr, recv, arg, "then_some");
                },
                (sym::try_into, []) if is_trait_method(cx, expr, sym::TryInto) => {
                    unnecessary_fallible_conversions::check_method(cx, expr);
                },
                (sym::to_owned, []) => {
                    if !suspicious_to_owned::check(cx, expr, recv) {
                        implicit_clone::check(cx, name, expr, recv);
                    }
                },
                (sym::to_os_string | sym::to_path_buf | sym::to_vec, []) => {
                    implicit_clone::check(cx, name, expr, recv);
                },
                (sym::type_id, []) => {
                    type_id_on_box::check(cx, recv, expr.span);
                },
                (sym::unwrap, []) => {
                    match method_call(recv) {
                        Some((sym::get, recv, [get_arg], _, _)) => {
                            get_unwrap::check(cx, expr, recv, get_arg, false);
                        },
                        Some((sym::get_mut, recv, [get_arg], _, _)) => {
                            get_unwrap::check(cx, expr, recv, get_arg, true);
                        },
                        Some((sym::or, recv, [or_arg], or_span, _)) => {
                            or_then_unwrap::check(cx, expr, recv, or_arg, or_span);
                        },
                        _ => {},
                    }
                    unnecessary_literal_unwrap::check(cx, expr, recv, name, args);
                },
                (sym::unwrap_or, [u_arg]) => {
                    match method_call(recv) {
                        Some((arith @ (sym::checked_add | sym::checked_sub | sym::checked_mul), lhs, [rhs], _, _)) => {
                            manual_saturating_arithmetic::check(
                                cx,
                                expr,
                                lhs,
                                rhs,
                                u_arg,
                                &arith.as_str()[const { "checked_".len() }..],
                            );
                        },
                        Some((sym::map, m_recv, [m_arg], span, _)) => {
                            option_map_unwrap_or::check(cx, expr, m_recv, m_arg, recv, u_arg, span, self.msrv);
                        },
                        Some((then_method @ (sym::then | sym::then_some), t_recv, [t_arg], _, _)) => {
                            obfuscated_if_else::check(cx, expr, t_recv, t_arg, Some(u_arg), then_method, name);
                        },
                        _ => {},
                    }
                    unnecessary_literal_unwrap::check(cx, expr, recv, name, args);
                },
                (sym::unwrap_or_default, []) => {
                    match method_call(recv) {
                        Some((sym::map, m_recv, [arg], span, _)) => {
                            manual_is_variant_and::check(cx, expr, m_recv, arg, span, self.msrv);
                        },
                        Some((then_method @ (sym::then | sym::then_some), t_recv, [t_arg], _, _)) => {
                            obfuscated_if_else::check(
                                cx,
                                expr,
                                t_recv,
                                t_arg,
                                None,
                                then_method,
                                sym::unwrap_or_default,
                            );
                        },
                        _ => {},
                    }
                    unnecessary_literal_unwrap::check(cx, expr, recv, name, args);
                },
                (sym::unwrap_or_else, [u_arg]) => {
                    match method_call(recv) {
                        Some((sym::map, recv, [map_arg], _, _))
                            if map_unwrap_or::check(cx, expr, recv, map_arg, u_arg, self.msrv) => {},
                        Some((then_method @ (sym::then | sym::then_some), t_recv, [t_arg], _, _)) => {
                            obfuscated_if_else::check(
                                cx,
                                expr,
                                t_recv,
                                t_arg,
                                Some(u_arg),
                                then_method,
                                sym::unwrap_or_else,
                            );
                        },
                        _ => {
                            unnecessary_lazy_eval::check(cx, expr, recv, u_arg, "unwrap_or");
                        },
                    }
                    unnecessary_literal_unwrap::check(cx, expr, recv, name, args);
                },
                (sym::wake, []) => {
                    waker_clone_wake::check(cx, expr, recv);
                },
                (sym::write, []) => {
                    readonly_write_lock::check(cx, expr, recv);
                },
                (sym::zip, [arg]) => {
                    if let ExprKind::MethodCall(name, iter_recv, [], _) = recv.kind
                        && name.ident.name == sym::iter
                    {
                        range_zip_with_len::check(cx, expr, iter_recv, arg);
                    }
                },
                _ => {},
            }
        }
        // Handle method calls whose receiver and arguments may come from expansion
        if let ExprKind::MethodCall(path, recv, args, _call_span) = expr.kind {
            match (path.ident.name, args) {
                (sym::expect, [_]) if !matches!(method_call(recv), Some((sym::ok | sym::err, _, [], _, _))) => {
                    unwrap_expect_used::check(
                        cx,
                        expr,
                        recv,
                        false,
                        self.allow_expect_in_consts,
                        self.allow_expect_in_tests,
                        unwrap_expect_used::Variant::Expect,
                    );
                },
                (sym::expect_err, [_]) => {
                    unwrap_expect_used::check(
                        cx,
                        expr,
                        recv,
                        true,
                        self.allow_expect_in_consts,
                        self.allow_expect_in_tests,
                        unwrap_expect_used::Variant::Expect,
                    );
                },
                (sym::unwrap, []) => {
                    unwrap_expect_used::check(
                        cx,
                        expr,
                        recv,
                        false,
                        self.allow_unwrap_in_consts,
                        self.allow_unwrap_in_tests,
                        unwrap_expect_used::Variant::Unwrap,
                    );
                },
                (sym::unwrap_err, []) => {
                    unwrap_expect_used::check(
                        cx,
                        expr,
                        recv,
                        true,
                        self.allow_unwrap_in_consts,
                        self.allow_unwrap_in_tests,
                        unwrap_expect_used::Variant::Unwrap,
                    );
                },
                _ => {},
            }
        }
    }
}

fn check_is_some_is_none(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, call_span: Span, is_some: bool) {
    match method_call(recv) {
        Some((name @ (sym::find | sym::position | sym::rposition), f_recv, [arg], span, _)) => {
            search_is_some::check(cx, expr, name, is_some, f_recv, arg, recv, span);
        },
        Some((sym::get, f_recv, [arg], _, _)) => {
            unnecessary_get_then_check::check(cx, call_span, recv, f_recv, arg, is_some);
        },
        Some((sym::first, f_recv, [], _, _)) => {
            unnecessary_first_then_check::check(cx, call_span, recv, f_recv, is_some);
        },
        _ => {},
    }
}

/// Used for `lint_binary_expr_with_method_call`.
#[derive(Copy, Clone)]
struct BinaryExprInfo<'a> {
    expr: &'a Expr<'a>,
    chain: &'a Expr<'a>,
    other: &'a Expr<'a>,
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
    safety: hir::HeaderSafety::Normal(hir::Safety::Safe),
    constness: hir::Constness::NotConst,
    asyncness: hir::IsAsync::NotAsync,
    abi: ExternAbi::Rust,
};

struct ShouldImplTraitCase {
    trait_name: &'static str,
    method_name: Symbol,
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
        method_name: Symbol,
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
    ShouldImplTraitCase::new("std::ops::Add", sym::add,  2,  FN_HEADER,  SelfKind::Value,  OutType::Any, true),
    ShouldImplTraitCase::new("std::convert::AsMut", sym::as_mut,  1,  FN_HEADER,  SelfKind::RefMut,  OutType::Ref, true),
    ShouldImplTraitCase::new("std::convert::AsRef", sym::as_ref,  1,  FN_HEADER,  SelfKind::Ref,  OutType::Ref, true),
    ShouldImplTraitCase::new("std::ops::BitAnd", sym::bitand,  2,  FN_HEADER,  SelfKind::Value,  OutType::Any, true),
    ShouldImplTraitCase::new("std::ops::BitOr", sym::bitor,  2,  FN_HEADER,  SelfKind::Value,  OutType::Any, true),
    ShouldImplTraitCase::new("std::ops::BitXor", sym::bitxor,  2,  FN_HEADER,  SelfKind::Value,  OutType::Any, true),
    ShouldImplTraitCase::new("std::borrow::Borrow", sym::borrow,  1,  FN_HEADER,  SelfKind::Ref,  OutType::Ref, true),
    ShouldImplTraitCase::new("std::borrow::BorrowMut", sym::borrow_mut,  1,  FN_HEADER,  SelfKind::RefMut,  OutType::Ref, true),
    ShouldImplTraitCase::new("std::clone::Clone", sym::clone,  1,  FN_HEADER,  SelfKind::Ref,  OutType::Any, true),
    ShouldImplTraitCase::new("std::cmp::Ord", sym::cmp,  2,  FN_HEADER,  SelfKind::Ref,  OutType::Any, true),
    ShouldImplTraitCase::new("std::default::Default", kw::Default,  0,  FN_HEADER,  SelfKind::No,  OutType::Any, true),
    ShouldImplTraitCase::new("std::ops::Deref", sym::deref,  1,  FN_HEADER,  SelfKind::Ref,  OutType::Ref, true),
    ShouldImplTraitCase::new("std::ops::DerefMut", sym::deref_mut,  1,  FN_HEADER,  SelfKind::RefMut,  OutType::Ref, true),
    ShouldImplTraitCase::new("std::ops::Div", sym::div,  2,  FN_HEADER,  SelfKind::Value,  OutType::Any, true),
    ShouldImplTraitCase::new("std::ops::Drop", sym::drop,  1,  FN_HEADER,  SelfKind::RefMut,  OutType::Unit, true),
    ShouldImplTraitCase::new("std::cmp::PartialEq", sym::eq,  2,  FN_HEADER,  SelfKind::Ref,  OutType::Bool, true),
    ShouldImplTraitCase::new("std::iter::FromIterator", sym::from_iter,  1,  FN_HEADER,  SelfKind::No,  OutType::Any, true),
    ShouldImplTraitCase::new("std::str::FromStr", sym::from_str,  1,  FN_HEADER,  SelfKind::No,  OutType::Any, true),
    ShouldImplTraitCase::new("std::hash::Hash", sym::hash,  2,  FN_HEADER,  SelfKind::Ref,  OutType::Unit, true),
    ShouldImplTraitCase::new("std::ops::Index", sym::index,  2,  FN_HEADER,  SelfKind::Ref,  OutType::Ref, true),
    ShouldImplTraitCase::new("std::ops::IndexMut", sym::index_mut,  2,  FN_HEADER,  SelfKind::RefMut,  OutType::Ref, true),
    ShouldImplTraitCase::new("std::iter::IntoIterator", sym::into_iter,  1,  FN_HEADER,  SelfKind::Value,  OutType::Any, true),
    ShouldImplTraitCase::new("std::ops::Mul", sym::mul,  2,  FN_HEADER,  SelfKind::Value,  OutType::Any, true),
    ShouldImplTraitCase::new("std::ops::Neg", sym::neg,  1,  FN_HEADER,  SelfKind::Value,  OutType::Any, true),
    ShouldImplTraitCase::new("std::iter::Iterator", sym::next,  1,  FN_HEADER,  SelfKind::RefMut,  OutType::Any, false),
    ShouldImplTraitCase::new("std::ops::Not", sym::not,  1,  FN_HEADER,  SelfKind::Value,  OutType::Any, true),
    ShouldImplTraitCase::new("std::ops::Rem", sym::rem,  2,  FN_HEADER,  SelfKind::Value,  OutType::Any, true),
    ShouldImplTraitCase::new("std::ops::Shl", sym::shl,  2,  FN_HEADER,  SelfKind::Value,  OutType::Any, true),
    ShouldImplTraitCase::new("std::ops::Shr", sym::shr,  2,  FN_HEADER,  SelfKind::Value,  OutType::Any, true),
    ShouldImplTraitCase::new("std::ops::Sub", sym::sub,  2,  FN_HEADER,  SelfKind::Value,  OutType::Any, true),
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
            } else if let Some(boxed_ty) = ty.boxed_ty() {
                boxed_ty == parent_ty
            } else if is_type_diagnostic_item(cx, ty, sym::Rc) || is_type_diagnostic_item(cx, ty, sym::Arc) {
                if let ty::Adt(_, args) = ty.kind() {
                    args.types().next() == Some(parent_ty)
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
                return false;
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
    expected.constness == actual.constness && expected.safety == actual.safety && expected.asyncness == actual.asyncness
}
