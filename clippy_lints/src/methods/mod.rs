mod bind_instead_of_map;
mod bytes_nth;
mod chars_cmp;
mod chars_cmp_with_unwrap;
mod chars_last_cmp;
mod chars_last_cmp_with_unwrap;
mod chars_next_cmp;
mod chars_next_cmp_with_unwrap;
mod clone_on_copy;
mod clone_on_ref_ptr;
mod cloned_instead_of_copied;
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
mod get_unwrap;
mod implicit_clone;
mod inefficient_to_string;
mod inspect_for_each;
mod into_iter_on_ref;
mod iter_cloned_collect;
mod iter_count;
mod iter_next_slice;
mod iter_nth;
mod iter_nth_zero;
mod iter_skip_next;
mod iterator_step_by_zero;
mod manual_saturating_arithmetic;
mod manual_split_once;
mod manual_str_repeat;
mod map_collect_result_unit;
mod map_flatten;
mod map_identity;
mod map_unwrap_or;
mod ok_expect;
mod option_as_ref_deref;
mod option_map_or_none;
mod option_map_unwrap_or;
mod or_fun_call;
mod search_is_some;
mod single_char_add_str;
mod single_char_insert_string;
mod single_char_pattern;
mod single_char_push_string;
mod skip_while_next;
mod string_extend_chars;
mod suspicious_map;
mod suspicious_splitn;
mod uninit_assumed_init;
mod unnecessary_filter_map;
mod unnecessary_fold;
mod unnecessary_lazy_eval;
mod unwrap_or_else_default;
mod unwrap_used;
mod useless_asref;
mod utils;
mod wrong_self_convention;
mod zst_offset;

use bind_instead_of_map::BindInsteadOfMap;
use clippy_utils::consts::{constant, Constant};
use clippy_utils::diagnostics::{span_lint, span_lint_and_help};
use clippy_utils::ty::{contains_adt_constructor, contains_ty, implements_trait, is_copy, is_type_diagnostic_item};
use clippy_utils::{contains_return, get_trait_def_id, in_macro, iter_input_pats, meets_msrv, msrvs, paths, return_ty};
use if_chain::if_chain;
use rustc_hir as hir;
use rustc_hir::def::Res;
use rustc_hir::{Expr, ExprKind, PrimTy, QPath, TraitItem, TraitItemKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::{self, TraitRef, Ty, TyS};
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::symbol::SymbolStr;
use rustc_span::{sym, Span};
use rustc_typeck::hir_ty_to_ty;

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
    pub CLONED_INSTEAD_OF_COPIED,
    pedantic,
    "used `cloned` where `copied` could be used instead"
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
    pub FLAT_MAP_OPTION,
    pedantic,
    "used `flat_map` where `filter_map` could be used instead"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `.unwrap()` calls on `Option`s and on `Result`s.
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
    /// # let opt = Some(1);
    ///
    /// // Bad
    /// opt.unwrap();
    ///
    /// // Good
    /// opt.expect("more helpful message");
    /// ```
    ///
    /// // or
    ///
    /// ```rust
    /// # let res: Result<usize, ()> = Ok(1);
    ///
    /// // Bad
    /// res.unwrap();
    ///
    /// // Good
    /// res.expect("more helpful message");
    /// ```
    pub UNWRAP_USED,
    restriction,
    "using `.unwrap()` on `Result` or `Option`, which should at least get a better message using `expect()`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `.expect()` calls on `Option`s and `Result`s.
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
    /// # let opt = Some(1);
    ///
    /// // Bad
    /// opt.expect("one");
    ///
    /// // Good
    /// let opt = Some(1);
    /// opt?;
    /// ```
    ///
    /// // or
    ///
    /// ```rust
    /// # let res: Result<usize, ()> = Ok(1);
    ///
    /// // Bad
    /// res.expect("one");
    ///
    /// // Good
    /// res?;
    /// # Ok::<(), ()>(())
    /// ```
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
    pub SHOULD_IMPLEMENT_TRAIT,
    style,
    "defining a method that should be implementing a std trait"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for methods with certain name prefixes and which
    /// doesn't match how self is taken. The actual rules are:
    ///
    /// |Prefix |Postfix     |`self` taken           | `self` type  |
    /// |-------|------------|-----------------------|--------------|
    /// |`as_`  | none       |`&self` or `&mut self` | any          |
    /// |`from_`| none       | none                  | any          |
    /// |`into_`| none       |`self`                 | any          |
    /// |`is_`  | none       |`&self` or none        | any          |
    /// |`to_`  | `_mut`     |`&mut self`            | any          |
    /// |`to_`  | not `_mut` |`self`                 | `Copy`       |
    /// |`to_`  | not `_mut` |`&self`                | not `Copy`   |
    ///
    /// Note: Clippy doesn't trigger methods with `to_` prefix in:
    /// - Traits definition.
    /// Clippy can not tell if a type that implements a trait is `Copy` or not.
    /// - Traits implementation, when `&self` is taken.
    /// The method signature is controlled by the trait and often `&self` is required for all types that implement the trait
    /// (see e.g. the `std::string::ToString` trait).
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
    ///
    /// // Bad
    /// x.ok().expect("why did I do this again?");
    ///
    /// // Good
    /// x.expect("why did I do this again?");
    /// ```
    pub OK_EXPECT,
    style,
    "using `ok().expect()`, which gives worse error messages than calling `expect` directly on the Result"
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
    ///
    /// // Bad
    /// x.unwrap_or_else(Default::default);
    /// x.unwrap_or_else(u32::default);
    ///
    /// // Good
    /// x.unwrap_or_default();
    /// ```
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
    /// # let x = Some(1);
    ///
    /// // Bad
    /// x.map(|a| a + 1).unwrap_or(0);
    ///
    /// // Good
    /// x.map_or(0, |a| a + 1);
    /// ```
    ///
    /// // or
    ///
    /// ```rust
    /// # let x: Result<usize, ()> = Ok(1);
    /// # fn some_function(foo: ()) -> usize { 1 }
    ///
    /// // Bad
    /// x.map(|a| a + 1).unwrap_or_else(some_function);
    ///
    /// // Good
    /// x.map_or_else(some_function, |a| a + 1);
    /// ```
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
    ///
    /// // Bad
    /// opt.map_or(None, |a| Some(a + 1));
    ///
    /// // Good
    /// opt.and_then(|a| Some(a + 1));
    /// ```
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
    /// Bad:
    /// ```rust
    /// # let r: Result<u32, &str> = Ok(1);
    /// assert_eq!(Some(1), r.map_or(None, Some));
    /// ```
    ///
    /// Good:
    /// ```rust
    /// # let r: Result<u32, &str> = Ok(1);
    /// assert_eq!(Some(1), r.ok());
    /// ```
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
    /// Could be written as
    /// ```rust
    /// # let vec = vec![1];
    /// vec.iter().find(|x| **x == 0);
    /// ```
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
    /// Could be written as
    /// ```rust
    /// # let vec = vec![1];
    /// vec.iter().find(|x| **x != 0);
    /// ```
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
    /// `_.flat_map(_)`
    ///
    /// ### Example
    /// ```rust
    /// let vec = vec![vec![1]];
    ///
    /// // Bad
    /// vec.iter().map(|x| x.iter()).flatten();
    ///
    /// // Good
    /// vec.iter().flat_map(|x| x.iter());
    /// ```
    pub MAP_FLATTEN,
    pedantic,
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
    /// Bad:
    /// ```rust
    /// (0_i32..10)
    ///     .filter(|n| n.checked_add(1).is_some())
    ///     .map(|n| n.checked_add(1).unwrap());
    /// ```
    ///
    /// Good:
    /// ```rust
    /// (0_i32..10).filter_map(|n| n.checked_add(1));
    /// ```
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
    /// Bad:
    /// ```rust
    /// (0_i32..10)
    ///     .find(|n| n.checked_add(1).is_some())
    ///     .map(|n| n.checked_add(1).unwrap());
    /// ```
    ///
    /// Good:
    /// ```rust
    /// (0_i32..10).find_map(|n| n.checked_add(1));
    /// ```
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
    /// let vec = vec![1];
    /// vec.iter().find(|x| **x == 0).is_some();
    ///
    /// let _ = "hello world".find("world").is_none();
    /// ```
    /// Could be written as
    /// ```rust
    /// let vec = vec![1];
    /// vec.iter().any(|x| *x == 0);
    ///
    /// let _ = !"hello world".contains("world");
    /// ```
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
    /// Could be written as
    /// ```rust
    /// let name = "foo";
    /// if name.starts_with('_') {};
    /// ```
    pub CHARS_NEXT_CMP,
    style,
    "using `.chars().next()` to check if a string starts with a char"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to `.or(foo(..))`, `.unwrap_or(foo(..))`,
    /// etc., and suggests to use `or_else`, `unwrap_or_else`, etc., or
    /// `unwrap_or_default` instead.
    ///
    /// ### Why is this bad?
    /// The function will always be called and potentially
    /// allocate an object acting as the default.
    ///
    /// ### Known problems
    /// If the function has side-effects, not calling it will
    /// change the semantic of the program, but you shouldn't rely on that anyway.
    ///
    /// ### Example
    /// ```rust
    /// # let foo = Some(String::new());
    /// foo.unwrap_or(String::new());
    /// ```
    /// this can instead be written:
    /// ```rust
    /// # let foo = Some(String::new());
    /// foo.unwrap_or_else(String::new);
    /// ```
    /// or
    /// ```rust
    /// # let foo = Some(String::new());
    /// foo.unwrap_or_default();
    /// ```
    pub OR_FUN_CALL,
    perf,
    "using any `*or` method with a function call, which suggests `*or_else`"
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
    /// ```
    /// or
    /// ```rust
    /// # let foo = Some(String::new());
    /// # let err_code = "418";
    /// # let err_msg = "I'm a teapot";
    /// foo.expect(format!("Err {}: {}", err_code, err_msg).as_str());
    /// ```
    /// this can instead be written:
    /// ```rust
    /// # let foo = Some(String::new());
    /// # let err_code = "418";
    /// # let err_msg = "I'm a teapot";
    /// foo.unwrap_or_else(|| panic!("Err {}: {}", err_code, err_msg));
    /// ```
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
    /// // Bad
    /// x.clone();
    ///
    /// // Good
    /// Rc::clone(&x);
    /// ```
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
    /// // Bad
    /// _.split("x");
    ///
    /// // Good
    /// _.split('x');
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
    /// // Bad
    /// # let mut s = HashSet::new();
    /// # s.insert(1);
    /// let x = s.iter().nth(0);
    ///
    /// // Good
    /// # let mut s = HashSet::new();
    /// # s.insert(1);
    /// let x = s.iter().next();
    /// ```
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
    pub ITER_SKIP_NEXT,
    style,
    "using `.skip(x).next()` on an iterator"
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
    /// // Bad
    /// a.extend(b.drain(..));
    ///
    /// // Good
    /// a.append(&mut b);
    /// ```
    pub EXTEND_WITH_DRAIN,
    perf,
    "using vec.append(&mut vec) to move the full range of a vecor to another"
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
    ///
    /// // Bad
    /// name.chars().last() == Some('_') || name.chars().next_back() == Some('-');
    ///
    /// // Good
    /// name.ends_with('_') || name.ends_with('-');
    /// ```
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
    /// let _ = (0..3).fold(false, |acc, x| acc || x > 2);
    /// ```
    /// This could be written as:
    /// ```rust
    /// let _ = (0..3).any(|x| x > 2);
    /// ```
    pub UNNECESSARY_FOLD,
    style,
    "using `fold` when a more succinct alternative exists"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `filter_map` calls which could be replaced by `filter` or `map`.
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
    pub UNNECESSARY_FILTER_MAP,
    complexity,
    "using `filter_map` when a more succinct alternative exists"
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
    /// // Bad
    /// let _ = (&vec![3, 4, 5]).into_iter();
    ///
    /// // Good
    /// let _ = (&vec![3, 4, 5]).iter();
    /// ```
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
    /// If the `map` call is intentional, this should be rewritten. Or, if you intend to
    /// drive the iterator to completion, you can just use `for_each` instead.
    ///
    /// ### Example
    /// ```rust
    /// let _ = (0..3).map(|x| x + 2).count();
    /// ```
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
    pub MANUAL_SATURATING_ARITHMETIC,
    style,
    "`.chcked_add/sub(x).unwrap_or(MAX/MIN)`"
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
    pub FILETYPE_IS_FILE,
    restriction,
    "`FileType::is_file` is not recommended to test for readable file type"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `_.as_ref().map(Deref::deref)` or it's aliases (such as String::as_str).
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
    /// let mut string = String::new();
    /// string.insert_str(0, "R");
    /// string.push_str("R");
    /// ```
    /// Could be written as
    /// ```rust
    /// let mut string = String::new();
    /// string.insert(0, 'R');
    /// string.push('R');
    /// ```
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
    /// use std::iter::FromIterator;
    ///
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
    /// // Bad
    /// let _ = "Hello".bytes().nth(3);
    ///
    /// // Good
    /// let _ = "Hello".as_bytes().get(3);
    /// ```
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
    /// // Bad
    /// let some_vec = vec![0, 1, 2, 3];
    /// let _ = some_vec.iter().count();
    /// let _ = &some_vec[..].iter().count();
    ///
    /// // Good
    /// let some_vec = vec![0, 1, 2, 3];
    /// let _ = some_vec.len();
    /// let _ = &some_vec[..].len();
    /// ```
    pub ITER_COUNT,
    complexity,
    "replace `.iter().count()` with `.len()`"
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
    /// // Bad
    /// let s = "";
    /// for x in s.splitn(1, ":") {
    ///     // use x
    /// }
    ///
    /// // Good
    /// let s = "";
    /// for x in s.splitn(2, ":") {
    ///     // use x
    /// }
    /// ```
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
    /// // Bad
    /// let x: String = std::iter::repeat('x').take(10).collect();
    ///
    /// // Good
    /// let x: String = "x".repeat(10);
    /// ```
    pub MANUAL_STR_REPEAT,
    perf,
    "manual implementation of `str::repeat`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for usages of `str::splitn(2, _)`
    ///
    /// **Why is this bad?** `split_once` is both clearer in intent and slightly more efficient.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust,ignore
    /// // Bad
    ///  let (key, value) = _.splitn(2, '=').next_tuple()?;
    ///  let value = _.splitn(2, '=').nth(1)?;
    ///
    /// // Good
    /// let (key, value) = _.split_once('=')?;
    /// let value = _.split_once('=')?.1;
    /// ```
    pub MANUAL_SPLIT_ONCE,
    complexity,
    "replace `.splitn(2, pat)` with `.split_once(pat)`"
}

pub struct Methods {
    avoid_breaking_exported_api: bool,
    msrv: Option<RustcVersion>,
}

impl Methods {
    #[must_use]
    pub fn new(avoid_breaking_exported_api: bool, msrv: Option<RustcVersion>) -> Self {
        Self {
            avoid_breaking_exported_api,
            msrv,
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
    EXPECT_FUN_CALL,
    CHARS_NEXT_CMP,
    CHARS_LAST_CMP,
    CLONE_ON_COPY,
    CLONE_ON_REF_PTR,
    CLONE_DOUBLE_REF,
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
    STRING_EXTEND_CHARS,
    ITER_CLONED_COLLECT,
    USELESS_ASREF,
    UNNECESSARY_FOLD,
    UNNECESSARY_FILTER_MAP,
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
    SUSPICIOUS_SPLITN,
    MANUAL_STR_REPEAT,
    EXTEND_WITH_DRAIN,
    MANUAL_SPLIT_ONCE
]);

/// Extracts a method call name, args, and `Span` of the method name.
fn method_call<'tcx>(recv: &'tcx hir::Expr<'tcx>) -> Option<(SymbolStr, &'tcx [hir::Expr<'tcx>], Span)> {
    if let ExprKind::MethodCall(path, span, args, _) = recv.kind {
        if !args.iter().any(|e| e.span.from_expansion()) {
            return Some((path.ident.name.as_str(), args, span));
        }
    }
    None
}

/// Same as `method_call` but the `SymbolStr` is dereferenced into a temporary `&str`
macro_rules! method_call {
    ($expr:expr) => {
        method_call($expr)
            .as_ref()
            .map(|&(ref name, args, span)| (&**name, args, span))
    };
}

impl<'tcx> LateLintPass<'tcx> for Methods {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if in_macro(expr.span) {
            return;
        }

        check_methods(cx, expr, self.msrv.as_ref());

        match expr.kind {
            hir::ExprKind::Call(func, args) => {
                from_iter_instead_of_collect::check(cx, expr, args, func);
            },
            hir::ExprKind::MethodCall(method_call, ref method_span, args, _) => {
                or_fun_call::check(cx, expr, *method_span, &method_call.ident.as_str(), args);
                expect_fun_call::check(cx, expr, *method_span, &method_call.ident.as_str(), args);
                clone_on_copy::check(cx, expr, method_call.ident.name, args);
                clone_on_ref_ptr::check(cx, expr, method_call.ident.name, args);
                inefficient_to_string::check(cx, expr, method_call.ident.name, args);
                single_char_add_str::check(cx, expr, args);
                into_iter_on_ref::check(cx, expr, *method_span, method_call.ident.name, args);
                single_char_pattern::check(cx, expr, method_call.ident.name, args);
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
        let parent = cx.tcx.hir().get_parent_item(impl_item.hir_id());
        let item = cx.tcx.hir().expect_item(parent);
        let self_ty = cx.tcx.type_of(item.def_id);

        let implements_trait = matches!(item.kind, hir::ItemKind::Impl(hir::Impl { of_trait: Some(_), .. }));
        if_chain! {
            if let hir::ImplItemKind::Fn(ref sig, id) = impl_item.kind;
            if let Some(first_arg) = iter_input_pats(sig.decl, cx.tcx.hir().body(id)).next();

            let method_sig = cx.tcx.fn_sig(impl_item.def_id);
            let method_sig = cx.tcx.erase_late_bound_regions(method_sig);

            let first_arg_ty = &method_sig.inputs().iter().next();

            // check conventions w.r.t. conversion method names and predicates
            if let Some(first_arg_ty) = first_arg_ty;

            then {
                // if this impl block implements a trait, lint in trait definition instead
                if !implements_trait && cx.access_levels.is_exported(impl_item.def_id) {
                    // check missing trait implementations
                    for method_config in &TRAIT_METHODS {
                        if name == method_config.method_name &&
                            sig.decl.inputs.len() == method_config.param_count &&
                            method_config.output_type.matches(&sig.decl.output) &&
                            method_config.self_kind.matches(cx, self_ty, first_arg_ty) &&
                            fn_header_equals(method_config.fn_header, sig.header) &&
                            method_config.lifetime_param_cond(impl_item)
                        {
                            span_lint_and_help(
                                cx,
                                SHOULD_IMPLEMENT_TRAIT,
                                impl_item.span,
                                &format!(
                                    "method `{}` can be confused for the standard trait method `{}::{}`",
                                    method_config.method_name,
                                    method_config.trait_name,
                                    method_config.method_name
                                ),
                                None,
                                &format!(
                                    "consider implementing the trait `{}` or choosing a less ambiguous method name",
                                    method_config.trait_name
                                )
                            );
                        }
                    }
                }

                if sig.decl.implicit_self.has_implicit_self()
                    && !(self.avoid_breaking_exported_api
                        && cx.access_levels.is_exported(impl_item.def_id))
                {
                    wrong_self_convention::check(
                        cx,
                        &name,
                        self_ty,
                        first_arg_ty,
                        first_arg.pat.span,
                        implements_trait,
                        false
                    );
                }
            }
        }

        // if this impl block implements a trait, lint in trait definition instead
        if implements_trait {
            return;
        }

        if let hir::ImplItemKind::Fn(_, _) = impl_item.kind {
            let ret_ty = return_ty(cx, impl_item.hir_id());

            // walk the return type and check for Self (this does not check associated types)
            if let Some(self_adt) = self_ty.ty_adt_def() {
                if contains_adt_constructor(cx.tcx, ret_ty, self_adt) {
                    return;
                }
            } else if contains_ty(cx.tcx, ret_ty, self_ty) {
                return;
            }

            // if return type is impl trait, check the associated types
            if let ty::Opaque(def_id, _) = *ret_ty.kind() {
                // one of the associated types must be Self
                for &(predicate, _span) in cx.tcx.explicit_item_bounds(def_id) {
                    if let ty::PredicateKind::Projection(projection_predicate) = predicate.kind().skip_binder() {
                        // walk the associated type and check for Self
                        if let Some(self_adt) = self_ty.ty_adt_def() {
                            if contains_adt_constructor(cx.tcx, projection_predicate.ty, self_adt) {
                                return;
                            }
                        } else if contains_ty(cx.tcx, projection_predicate.ty, self_ty) {
                            return;
                        }
                    }
                }
            }

            if name == "new" && !TyS::same_type(ret_ty, self_ty) {
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
                let self_ty = TraitRef::identity(cx.tcx, item.def_id.to_def_id()).self_ty().skip_binder();
                wrong_self_convention::check(
                    cx,
                    &item.ident.name.as_str(),
                    self_ty,
                    first_arg_ty,
                    first_arg_span,
                    false,
                    true
                );
            }
        }

        if_chain! {
            if item.ident.name == sym::new;
            if let TraitItemKind::Fn(_, _) = item.kind;
            let ret_ty = return_ty(cx, item.hir_id());
            let self_ty = TraitRef::identity(cx.tcx, item.def_id.to_def_id()).self_ty().skip_binder();
            if !contains_ty(cx.tcx, ret_ty, self_ty);

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

#[allow(clippy::too_many_lines)]
fn check_methods<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, msrv: Option<&RustcVersion>) {
    if let Some((name, [recv, args @ ..], span)) = method_call!(expr) {
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
            ("as_mut", []) => useless_asref::check(cx, expr, "as_mut", recv),
            ("as_ref", []) => useless_asref::check(cx, expr, "as_ref", recv),
            ("assume_init", []) => uninit_assumed_init::check(cx, expr, recv),
            ("cloned", []) => cloned_instead_of_copied::check(cx, expr, recv, span, msrv),
            ("collect", []) => match method_call!(recv) {
                Some(("cloned", [recv2], _)) => iter_cloned_collect::check(cx, expr, recv2),
                Some(("map", [m_recv, m_arg], _)) => {
                    map_collect_result_unit::check(cx, expr, m_recv, m_arg, recv);
                },
                Some(("take", [take_self_arg, take_arg], _)) => {
                    if meets_msrv(msrv, &msrvs::STR_REPEAT) {
                        manual_str_repeat::check(cx, expr, recv, take_self_arg, take_arg);
                    }
                },
                _ => {},
            },
            ("count", []) => match method_call!(recv) {
                Some((name @ ("into_iter" | "iter" | "iter_mut"), [recv2], _)) => {
                    iter_count::check(cx, expr, recv2, name);
                },
                Some(("map", [_, arg], _)) => suspicious_map::check(cx, expr, recv, arg),
                _ => {},
            },
            ("expect", [_]) => match method_call!(recv) {
                Some(("ok", [recv], _)) => ok_expect::check(cx, expr, recv),
                _ => expect_used::check(cx, expr, recv),
            },
            ("extend", [arg]) => {
                string_extend_chars::check(cx, expr, recv, arg);
                extend_with_drain::check(cx, expr, recv, arg);
            },
            ("filter_map", [arg]) => {
                unnecessary_filter_map::check(cx, expr, arg);
                filter_map_identity::check(cx, expr, arg, span);
            },
            ("flat_map", [arg]) => {
                flat_map_identity::check(cx, expr, arg, span);
                flat_map_option::check(cx, expr, arg, span);
            },
            ("flatten", []) => {
                if let Some(("map", [recv, map_arg], _)) = method_call!(recv) {
                    map_flatten::check(cx, expr, recv, map_arg);
                }
            },
            ("fold", [init, acc]) => unnecessary_fold::check(cx, expr, init, acc, span),
            ("for_each", [_]) => {
                if let Some(("inspect", [_, _], span2)) = method_call!(recv) {
                    inspect_for_each::check(cx, expr, span2);
                }
            },
            ("get_or_insert_with", [arg]) => unnecessary_lazy_eval::check(cx, expr, recv, arg, "get_or_insert"),
            ("is_file", []) => filetype_is_file::check(cx, expr, recv),
            ("is_none", []) => check_is_some_is_none(cx, expr, recv, false),
            ("is_some", []) => check_is_some_is_none(cx, expr, recv, true),
            ("map", [m_arg]) => {
                if let Some((name, [recv2, args @ ..], span2)) = method_call!(recv) {
                    match (name, args) {
                        ("as_mut", []) => option_as_ref_deref::check(cx, expr, recv2, m_arg, true, msrv),
                        ("as_ref", []) => option_as_ref_deref::check(cx, expr, recv2, m_arg, false, msrv),
                        ("filter", [f_arg]) => {
                            filter_map::check(cx, expr, recv2, f_arg, span2, recv, m_arg, span, false);
                        },
                        ("find", [f_arg]) => filter_map::check(cx, expr, recv2, f_arg, span2, recv, m_arg, span, true),
                        _ => {},
                    }
                }
                map_identity::check(cx, expr, recv, m_arg, span);
            },
            ("map_or", [def, map]) => option_map_or_none::check(cx, expr, recv, def, map),
            ("next", []) => {
                if let Some((name, [recv, args @ ..], _)) = method_call!(recv) {
                    match (name, args) {
                        ("filter", [arg]) => filter_next::check(cx, expr, recv, arg),
                        ("filter_map", [arg]) => filter_map_next::check(cx, expr, recv, arg, msrv),
                        ("iter", []) => iter_next_slice::check(cx, expr, recv),
                        ("skip", [arg]) => iter_skip_next::check(cx, expr, recv, arg),
                        ("skip_while", [_]) => skip_while_next::check(cx, expr),
                        _ => {},
                    }
                }
            },
            ("nth", [n_arg]) => match method_call!(recv) {
                Some(("bytes", [recv2], _)) => bytes_nth::check(cx, expr, recv2, n_arg),
                Some(("iter", [recv2], _)) => iter_nth::check(cx, expr, recv2, recv, n_arg, false),
                Some(("iter_mut", [recv2], _)) => iter_nth::check(cx, expr, recv2, recv, n_arg, true),
                _ => iter_nth_zero::check(cx, expr, recv, n_arg),
            },
            ("ok_or_else", [arg]) => unnecessary_lazy_eval::check(cx, expr, recv, arg, "ok_or"),
            ("or_else", [arg]) => {
                if !bind_instead_of_map::ResultOrElseErrInfo::check(cx, expr, recv, arg) {
                    unnecessary_lazy_eval::check(cx, expr, recv, arg, "or");
                }
            },
            ("splitn" | "rsplitn", [count_arg, pat_arg]) => {
                if let Some((Constant::Int(count), _)) = constant(cx, cx.typeck_results(), count_arg) {
                    suspicious_splitn::check(cx, name, expr, recv, count);
                    if count == 2 && meets_msrv(msrv, &msrvs::STR_SPLIT_ONCE) {
                        manual_split_once::check(cx, name, expr, recv, pat_arg);
                    }
                }
            },
            ("splitn_mut" | "rsplitn_mut", [count_arg, _]) => {
                if let Some((Constant::Int(count), _)) = constant(cx, cx.typeck_results(), count_arg) {
                    suspicious_splitn::check(cx, name, expr, recv, count);
                }
            },
            ("step_by", [arg]) => iterator_step_by_zero::check(cx, expr, arg),
            ("to_os_string" | "to_owned" | "to_path_buf" | "to_vec", []) => {
                implicit_clone::check(cx, name, expr, recv, span);
            },
            ("unwrap", []) => match method_call!(recv) {
                Some(("get", [recv, get_arg], _)) => get_unwrap::check(cx, expr, recv, get_arg, false),
                Some(("get_mut", [recv, get_arg], _)) => get_unwrap::check(cx, expr, recv, get_arg, true),
                _ => unwrap_used::check(cx, expr, recv),
            },
            ("unwrap_or", [u_arg]) => match method_call!(recv) {
                Some((arith @ ("checked_add" | "checked_sub" | "checked_mul"), [lhs, rhs], _)) => {
                    manual_saturating_arithmetic::check(cx, expr, lhs, rhs, u_arg, &arith["checked_".len()..]);
                },
                Some(("map", [m_recv, m_arg], span)) => {
                    option_map_unwrap_or::check(cx, expr, m_recv, m_arg, recv, u_arg, span);
                },
                _ => {},
            },
            ("unwrap_or_else", [u_arg]) => match method_call!(recv) {
                Some(("map", [recv, map_arg], _)) if map_unwrap_or::check(cx, expr, recv, map_arg, u_arg, msrv) => {},
                _ => {
                    unwrap_or_else_default::check(cx, expr, recv, u_arg);
                    unnecessary_lazy_eval::check(cx, expr, recv, u_arg, "unwrap_or");
                },
            },
            _ => {},
        }
    }
}

fn check_is_some_is_none(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, is_some: bool) {
    if let Some((name @ ("find" | "position" | "rposition"), [f_recv, arg], span)) = method_call!(recv) {
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
    // FIXME: default doesn't work
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

#[derive(Clone, Copy, PartialEq, Debug)]
enum SelfKind {
    Value,
    Ref,
    RefMut,
    No,
}

impl SelfKind {
    fn matches<'a>(self, cx: &LateContext<'a>, parent_ty: Ty<'a>, ty: Ty<'a>) -> bool {
        fn matches_value<'a>(cx: &LateContext<'a>, parent_ty: Ty<'_>, ty: Ty<'_>) -> bool {
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

            let trait_path = match mutability {
                hir::Mutability::Not => &paths::ASREF_TRAIT,
                hir::Mutability::Mut => &paths::ASMUT_TRAIT,
            };

            let trait_def_id = match get_trait_def_id(cx, trait_path) {
                Some(did) => did,
                None => return false,
            };
            implements_trait(cx, ty, trait_def_id, &[parent_ty.into()])
        }

        match self {
            Self::Value => matches_value(cx, parent_ty, ty),
            Self::Ref => matches_ref(cx, hir::Mutability::Not, parent_ty, ty) || ty == parent_ty && is_copy(cx, ty),
            Self::RefMut => matches_ref(cx, hir::Mutability::Mut, parent_ty, ty),
            Self::No => ty != parent_ty,
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
            (Self::Ref, &hir::FnRetTy::Return(ty)) => matches!(ty.kind, hir::TyKind::Rptr(_, _)),
            _ => false,
        }
    }
}

fn is_bool(ty: &hir::Ty<'_>) -> bool {
    if let hir::TyKind::Path(QPath::Resolved(_, path)) = ty.kind {
        matches!(path.res, Res::PrimTy(PrimTy::Bool))
    } else {
        false
    }
}

fn fn_header_equals(expected: hir::FnHeader, actual: hir::FnHeader) -> bool {
    expected.constness == actual.constness
        && expected.unsafety == actual.unsafety
        && expected.asyncness == actual.asyncness
}
