mod bind_instead_of_map;
mod inefficient_to_string;
mod manual_saturating_arithmetic;
mod option_map_unwrap_or;
mod unnecessary_filter_map;
mod unnecessary_lazy_eval;

use std::borrow::Cow;
use std::fmt;
use std::iter;

use bind_instead_of_map::BindInsteadOfMap;
use if_chain::if_chain;
use rustc_ast::ast;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::{TraitItem, TraitItemKind};
use rustc_lint::{LateContext, LateLintPass, Lint, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::{self, TraitRef, Ty, TyS};
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::source_map::Span;
use rustc_span::symbol::{sym, SymbolStr};

use crate::consts::{constant, Constant};
use crate::utils::eager_or_lazy::is_lazyness_candidate;
use crate::utils::usage::mutated_variables;
use crate::utils::{
    contains_return, contains_ty, get_arg_name, get_parent_expr, get_trait_def_id, has_iter_method, higher,
    implements_trait, in_macro, is_copy, is_expn_of, is_type_diagnostic_item, iter_input_pats, last_path_segment,
    match_def_path, match_qpath, match_trait_method, match_type, match_var, meets_msrv, method_calls,
    method_chain_args, paths, remove_blocks, return_ty, single_segment_path, snippet, snippet_with_applicability,
    snippet_with_macro_callsite, span_lint, span_lint_and_help, span_lint_and_sugg, span_lint_and_then, sugg,
    walk_ptrs_ty_depth, SpanlessEq,
};

declare_clippy_lint! {
    /// **What it does:** Checks for `.unwrap()` calls on `Option`s and on `Result`s.
    ///
    /// **Why is this bad?** It is better to handle the `None` or `Err` case,
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
    /// **Known problems:** None.
    ///
    /// **Examples:**
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
    /// **What it does:** Checks for `.expect()` calls on `Option`s and `Result`s.
    ///
    /// **Why is this bad?** Usually it is better to handle the `None` or `Err` case.
    /// Still, for a lot of quick-and-dirty code, `expect` is a good choice, which is why
    /// this lint is `Allow` by default.
    ///
    /// `result.expect()` will let the thread panic on `Err`
    /// values. Normally, you want to implement more sophisticated error handling,
    /// and propagate errors upwards with `?` operator.
    ///
    /// **Known problems:** None.
    ///
    /// **Examples:**
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
    /// **What it does:** Checks for methods that should live in a trait
    /// implementation of a `std` trait (see [llogiq's blog
    /// post](http://llogiq.github.io/2015/07/30/traits.html) for further
    /// information) instead of an inherent implementation.
    ///
    /// **Why is this bad?** Implementing the traits improve ergonomics for users of
    /// the code, often with very little cost. Also people seeing a `mul(...)`
    /// method
    /// may expect `*` to work equally, so you should have good reason to disappoint
    /// them.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
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
    /// **What it does:** Checks for methods with certain name prefixes and which
    /// doesn't match how self is taken. The actual rules are:
    ///
    /// |Prefix |`self` taken          |
    /// |-------|----------------------|
    /// |`as_`  |`&self` or `&mut self`|
    /// |`from_`| none                 |
    /// |`into_`|`self`                |
    /// |`is_`  |`&self` or none       |
    /// |`to_`  |`&self`               |
    ///
    /// **Why is this bad?** Consistency breeds readability. If you follow the
    /// conventions, your users won't be surprised that they, e.g., need to supply a
    /// mutable reference to a `as_..` function.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
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
    /// **What it does:** This is the same as
    /// [`wrong_self_convention`](#wrong_self_convention), but for public items.
    ///
    /// **Why is this bad?** See [`wrong_self_convention`](#wrong_self_convention).
    ///
    /// **Known problems:** Actually *renaming* the function may break clients if
    /// the function is part of the public interface. In that case, be mindful of
    /// the stability guarantees you've given your users.
    ///
    /// **Example:**
    /// ```rust
    /// # struct X;
    /// impl<'a> X {
    ///     pub fn as_str(self) -> &'a str {
    ///         "foo"
    ///     }
    /// }
    /// ```
    pub WRONG_PUB_SELF_CONVENTION,
    restriction,
    "defining a public method named with an established prefix (like \"into_\") that takes `self` with the wrong convention"
}

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `ok().expect(..)`.
    ///
    /// **Why is this bad?** Because you usually call `expect()` on the `Result`
    /// directly to get a better error message.
    ///
    /// **Known problems:** The error type needs to implement `Debug`
    ///
    /// **Example:**
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
    /// **What it does:** Checks for usage of `option.map(_).unwrap_or(_)` or `option.map(_).unwrap_or_else(_)` or
    /// `result.map(_).unwrap_or_else(_)`.
    ///
    /// **Why is this bad?** Readability, these can be written more concisely (resp.) as
    /// `option.map_or(_, _)`, `option.map_or_else(_, _)` and `result.map_or_else(_, _)`.
    ///
    /// **Known problems:** The order of the arguments is not in execution order
    ///
    /// **Examples:**
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
    /// **What it does:** Checks for usage of `_.map_or(None, _)`.
    ///
    /// **Why is this bad?** Readability, this can be written more concisely as
    /// `_.and_then(_)`.
    ///
    /// **Known problems:** The order of the arguments is not in execution order.
    ///
    /// **Example:**
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
    /// **What it does:** Checks for usage of `_.map_or(None, Some)`.
    ///
    /// **Why is this bad?** Readability, this can be written more concisely as
    /// `_.ok()`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
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
    /// **What it does:** Checks for usage of `_.and_then(|x| Some(y))`, `_.and_then(|x| Ok(y))` or
    /// `_.or_else(|x| Err(y))`.
    ///
    /// **Why is this bad?** Readability, this can be written more concisely as
    /// `_.map(|x| y)` or `_.map_err(|x| y)`.
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
    ///
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
    /// **What it does:** Checks for usage of `_.filter(_).next()`.
    ///
    /// **Why is this bad?** Readability, this can be written more concisely as
    /// `_.find(_)`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
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
    /// **What it does:** Checks for usage of `_.skip_while(condition).next()`.
    ///
    /// **Why is this bad?** Readability, this can be written more concisely as
    /// `_.find(!condition)`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
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
    /// **What it does:** Checks for usage of `_.map(_).flatten(_)`,
    ///
    /// **Why is this bad?** Readability, this can be written more concisely as
    /// `_.flat_map(_)`
    ///
    /// **Known problems:**
    ///
    /// **Example:**
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
    /// **What it does:** Checks for usage of `_.filter(_).map(_)`,
    /// `_.filter(_).flat_map(_)`, `_.filter_map(_).flat_map(_)` and similar.
    ///
    /// **Why is this bad?** Readability, this can be written more concisely as
    /// `_.filter_map(_)`.
    ///
    /// **Known problems:** Often requires a condition + Option/Iterator creation
    /// inside the closure.
    ///
    /// **Example:**
    /// ```rust
    /// let vec = vec![1];
    ///
    /// // Bad
    /// vec.iter().filter(|x| **x == 0).map(|x| *x * 2);
    ///
    /// // Good
    /// vec.iter().filter_map(|x| if *x == 0 {
    ///     Some(*x * 2)
    /// } else {
    ///     None
    /// });
    /// ```
    pub FILTER_MAP,
    pedantic,
    "using combinations of `filter`, `map`, `filter_map` and `flat_map` which can usually be written as a single method call"
}

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `_.filter_map(_).next()`.
    ///
    /// **Why is this bad?** Readability, this can be written more concisely as
    /// `_.find_map(_)`.
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
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
    /// **What it does:** Checks for usage of `flat_map(|x| x)`.
    ///
    /// **Why is this bad?** Readability, this can be written more concisely by using `flatten`.
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
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
    /// **What it does:** Checks for usage of `_.find(_).map(_)`.
    ///
    /// **Why is this bad?** Readability, this can be written more concisely as
    /// `_.find_map(_)`.
    ///
    /// **Known problems:** Often requires a condition + Option/Iterator creation
    /// inside the closure.
    ///
    /// **Example:**
    /// ```rust
    ///  (0..3).find(|x| *x == 2).map(|x| x * 2);
    /// ```
    /// Can be written as
    /// ```rust
    ///  (0..3).find_map(|x| if x == 2 { Some(x * 2) } else { None });
    /// ```
    pub FIND_MAP,
    pedantic,
    "using a combination of `find` and `map` can usually be written as a single method call"
}

declare_clippy_lint! {
    /// **What it does:** Checks for an iterator or string search (such as `find()`,
    /// `position()`, or `rposition()`) followed by a call to `is_some()`.
    ///
    /// **Why is this bad?** Readability, this can be written more concisely as
    /// `_.any(_)` or `_.contains(_)`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # let vec = vec![1];
    /// vec.iter().find(|x| **x == 0).is_some();
    /// ```
    /// Could be written as
    /// ```rust
    /// # let vec = vec![1];
    /// vec.iter().any(|x| *x == 0);
    /// ```
    pub SEARCH_IS_SOME,
    complexity,
    "using an iterator or string search followed by `is_some()`, which is more succinctly expressed as a call to `any()` or `contains()`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `.chars().next()` on a `str` to check
    /// if it starts with a given char.
    ///
    /// **Why is this bad?** Readability, this can be written more concisely as
    /// `_.starts_with(_)`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
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
    /// **What it does:** Checks for calls to `.or(foo(..))`, `.unwrap_or(foo(..))`,
    /// etc., and suggests to use `or_else`, `unwrap_or_else`, etc., or
    /// `unwrap_or_default` instead.
    ///
    /// **Why is this bad?** The function will always be called and potentially
    /// allocate an object acting as the default.
    ///
    /// **Known problems:** If the function has side-effects, not calling it will
    /// change the semantic of the program, but you shouldn't rely on that anyway.
    ///
    /// **Example:**
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
    /// **What it does:** Checks for calls to `.expect(&format!(...))`, `.expect(foo(..))`,
    /// etc., and suggests to use `unwrap_or_else` instead
    ///
    /// **Why is this bad?** The function will always be called.
    ///
    /// **Known problems:** If the function has side-effects, not calling it will
    /// change the semantics of the program, but you shouldn't rely on that anyway.
    ///
    /// **Example:**
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
    /// **What it does:** Checks for usage of `.clone()` on a `Copy` type.
    ///
    /// **Why is this bad?** The only reason `Copy` types implement `Clone` is for
    /// generics, not for using the `clone` method on a concrete type.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// 42u64.clone();
    /// ```
    pub CLONE_ON_COPY,
    complexity,
    "using `clone` on a `Copy` type"
}

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `.clone()` on a ref-counted pointer,
    /// (`Rc`, `Arc`, `rc::Weak`, or `sync::Weak`), and suggests calling Clone via unified
    /// function syntax instead (e.g., `Rc::clone(foo)`).
    ///
    /// **Why is this bad?** Calling '.clone()' on an Rc, Arc, or Weak
    /// can obscure the fact that only the pointer is being cloned, not the underlying
    /// data.
    ///
    /// **Example:**
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
    /// **What it does:** Checks for usage of `.clone()` on an `&&T`.
    ///
    /// **Why is this bad?** Cloning an `&&T` copies the inner `&T`, instead of
    /// cloning the underlying `T`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
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
    /// **What it does:** Checks for usage of `.to_string()` on an `&&T` where
    /// `T` implements `ToString` directly (like `&&str` or `&&String`).
    ///
    /// **Why is this bad?** This bypasses the specialized implementation of
    /// `ToString` and instead goes through the more expensive string formatting
    /// facilities.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
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
    /// **What it does:** Checks for `new` not returning a type that contains `Self`.
    ///
    /// **Why is this bad?** As a convention, `new` methods are used to make a new
    /// instance of a type.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
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
    /// **What it does:** Checks for string methods that receive a single-character
    /// `str` as an argument, e.g., `_.split("x")`.
    ///
    /// **Why is this bad?** Performing these methods using a `char` is faster than
    /// using a `str`.
    ///
    /// **Known problems:** Does not catch multi-byte unicode characters.
    ///
    /// **Example:**
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
    /// **What it does:** Checks for calling `.step_by(0)` on iterators which panics.
    ///
    /// **Why is this bad?** This very much looks like an oversight. Use `panic!()` instead if you
    /// actually intend to panic.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
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
    /// **What it does:** Checks for the use of `iter.nth(0)`.
    ///
    /// **Why is this bad?** `iter.next()` is equivalent to
    /// `iter.nth(0)`, as they both consume the next element,
    ///  but is more readable.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
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
    /// **What it does:** Checks for use of `.iter().nth()` (and the related
    /// `.iter_mut().nth()`) on standard library types with O(1) element access.
    ///
    /// **Why is this bad?** `.get()` and `.get_mut()` are more efficient and more
    /// readable.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
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
    /// **What it does:** Checks for use of `.skip(x).next()` on iterators.
    ///
    /// **Why is this bad?** `.nth(x)` is cleaner
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
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
    /// **What it does:** Checks for use of `.get().unwrap()` (or
    /// `.get_mut().unwrap`) on a standard library type which implements `Index`
    ///
    /// **Why is this bad?** Using the Index trait (`[]`) is more clear and more
    /// concise.
    ///
    /// **Known problems:** Not a replacement for error handling: Using either
    /// `.unwrap()` or the Index trait (`[]`) carries the risk of causing a `panic`
    /// if the value being accessed is `None`. If the use of `.get().unwrap()` is a
    /// temporary placeholder for dealing with the `Option` type, then this does
    /// not mitigate the need for error handling. If there is a chance that `.get()`
    /// will be `None` in your program, then it is advisable that the `None` case
    /// is handled in a future refactor instead of using `.unwrap()` or the Index
    /// trait.
    ///
    /// **Example:**
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
    /// **What it does:** Checks for the use of `.extend(s.chars())` where s is a
    /// `&str` or `String`.
    ///
    /// **Why is this bad?** `.push_str(s)` is clearer
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
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
    /// **What it does:** Checks for the use of `.cloned().collect()` on slice to
    /// create a `Vec`.
    ///
    /// **Why is this bad?** `.to_vec()` is clearer
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
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
    /// **What it does:** Checks for usage of `_.chars().last()` or
    /// `_.chars().next_back()` on a `str` to check if it ends with a given char.
    ///
    /// **Why is this bad?** Readability, this can be written more concisely as
    /// `_.ends_with(_)`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
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
    /// **What it does:** Checks for usage of `.as_ref()` or `.as_mut()` where the
    /// types before and after the call are the same.
    ///
    /// **Why is this bad?** The call is unnecessary.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
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
    /// **What it does:** Checks for using `fold` when a more succinct alternative exists.
    /// Specifically, this checks for `fold`s which could be replaced by `any`, `all`,
    /// `sum` or `product`.
    ///
    /// **Why is this bad?** Readability.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
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
    /// **What it does:** Checks for `filter_map` calls which could be replaced by `filter` or `map`.
    /// More specifically it checks if the closure provided is only performing one of the
    /// filter or map operations and suggests the appropriate option.
    ///
    /// **Why is this bad?** Complexity. The intent is also clearer if only a single
    /// operation is being performed.
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
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
    /// **What it does:** Checks for `into_iter` calls on references which should be replaced by `iter`
    /// or `iter_mut`.
    ///
    /// **Why is this bad?** Readability. Calling `into_iter` on a reference will not move out its
    /// content into the resulting iterator, which is confusing. It is better just call `iter` or
    /// `iter_mut` directly.
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
    ///
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
    /// **What it does:** Checks for calls to `map` followed by a `count`.
    ///
    /// **Why is this bad?** It looks suspicious. Maybe `map` was confused with `filter`.
    /// If the `map` call is intentional, this should be rewritten. Or, if you intend to
    /// drive the iterator to completion, you can just use `for_each` instead.
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
    ///
    /// ```rust
    /// let _ = (0..3).map(|x| x + 2).count();
    /// ```
    pub SUSPICIOUS_MAP,
    complexity,
    "suspicious usage of map"
}

declare_clippy_lint! {
    /// **What it does:** Checks for `MaybeUninit::uninit().assume_init()`.
    ///
    /// **Why is this bad?** For most types, this is undefined behavior.
    ///
    /// **Known problems:** For now, we accept empty tuples and tuples / arrays
    /// of `MaybeUninit`. There may be other types that allow uninitialized
    /// data, but those are not yet rigorously defined.
    ///
    /// **Example:**
    ///
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
    /// **What it does:** Checks for `.checked_add/sub(x).unwrap_or(MAX/MIN)`.
    ///
    /// **Why is this bad?** These can be written simply with `saturating_add/sub` methods.
    ///
    /// **Example:**
    ///
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
    /// **What it does:** Checks for `offset(_)`, `wrapping_`{`add`, `sub`}, etc. on raw pointers to
    /// zero-sized types
    ///
    /// **Why is this bad?** This is a no-op, and likely unintended
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
    /// ```rust
    /// unsafe { (&() as *const ()).offset(1) };
    /// ```
    pub ZST_OFFSET,
    correctness,
    "Check for offset calculations on raw pointers to zero-sized types"
}

declare_clippy_lint! {
    /// **What it does:** Checks for `FileType::is_file()`.
    ///
    /// **Why is this bad?** When people testing a file type with `FileType::is_file`
    /// they are testing whether a path is something they can get bytes from. But
    /// `is_file` doesn't cover special file types in unix-like systems, and doesn't cover
    /// symlink in windows. Using `!FileType::is_dir()` is a better way to that intention.
    ///
    /// **Example:**
    ///
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
    /// **What it does:** Checks for usage of `_.as_ref().map(Deref::deref)` or it's aliases (such as String::as_str).
    ///
    /// **Why is this bad?** Readability, this can be written more concisely as
    /// `_.as_deref()`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
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
    /// **What it does:** Checks for usage of `iter().next()` on a Slice or an Array
    ///
    /// **Why is this bad?** These can be shortened into `.get()`
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
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
    /// **What it does:** Warns when using `push_str`/`insert_str` with a single-character string literal
    /// where `push`/`insert` with a `char` would work fine.
    ///
    /// **Why is this bad?** It's less clear that we are pushing a single character.
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
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
    /// **What it does:** As the counterpart to `or_fun_call`, this lint looks for unnecessary
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
    /// **Why is this bad?** Using eager evaluation is shorter and simpler in some cases.
    ///
    /// **Known problems:** It is possible, but not recommended for `Deref` and `Index` to have
    /// side effects. Eagerly evaluating them can change the semantics of the program.
    ///
    /// **Example:**
    ///
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
    /// **What it does:** Checks for usage of `_.map(_).collect::<Result<(), _>()`.
    ///
    /// **Why is this bad?** Using `try_for_each` instead is more readable and idiomatic.
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
    ///
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
    /// **What it does:** Checks for `from_iter()` function calls on types that implement the `FromIterator`
    /// trait.
    ///
    /// **Why is this bad?** It is recommended style to use collect. See
    /// [FromIterator documentation](https://doc.rust-lang.org/std/iter/trait.FromIterator.html)
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
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
    style,
    "use `.collect()` instead of `::from_iter()`"
}

pub struct Methods {
    msrv: Option<RustcVersion>,
}

impl Methods {
    #[must_use]
    pub fn new(msrv: Option<RustcVersion>) -> Self {
        Self { msrv }
    }
}

impl_lint_pass!(Methods => [
    UNWRAP_USED,
    EXPECT_USED,
    SHOULD_IMPLEMENT_TRAIT,
    WRONG_SELF_CONVENTION,
    WRONG_PUB_SELF_CONVENTION,
    OK_EXPECT,
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
    INEFFICIENT_TO_STRING,
    NEW_RET_NO_SELF,
    SINGLE_CHAR_PATTERN,
    SINGLE_CHAR_ADD_STR,
    SEARCH_IS_SOME,
    FILTER_NEXT,
    SKIP_WHILE_NEXT,
    FILTER_MAP,
    FILTER_MAP_NEXT,
    FLAT_MAP_IDENTITY,
    FIND_MAP,
    MAP_FLATTEN,
    ITERATOR_STEP_BY_ZERO,
    ITER_NEXT_SLICE,
    ITER_NTH,
    ITER_NTH_ZERO,
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
]);

impl<'tcx> LateLintPass<'tcx> for Methods {
    #[allow(clippy::too_many_lines)]
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if in_macro(expr.span) {
            return;
        }

        let (method_names, arg_lists, method_spans) = method_calls(expr, 2);
        let method_names: Vec<SymbolStr> = method_names.iter().map(|s| s.as_str()).collect();
        let method_names: Vec<&str> = method_names.iter().map(|s| &**s).collect();

        match method_names.as_slice() {
            ["unwrap", "get"] => lint_get_unwrap(cx, expr, arg_lists[1], false),
            ["unwrap", "get_mut"] => lint_get_unwrap(cx, expr, arg_lists[1], true),
            ["unwrap", ..] => lint_unwrap(cx, expr, arg_lists[0]),
            ["expect", "ok"] => lint_ok_expect(cx, expr, arg_lists[1]),
            ["expect", ..] => lint_expect(cx, expr, arg_lists[0]),
            ["unwrap_or", "map"] => option_map_unwrap_or::lint(cx, expr, arg_lists[1], arg_lists[0], method_spans[1]),
            ["unwrap_or_else", "map"] => {
                if !lint_map_unwrap_or_else(cx, expr, arg_lists[1], arg_lists[0]) {
                    unnecessary_lazy_eval::lint(cx, expr, arg_lists[0], "unwrap_or");
                }
            },
            ["map_or", ..] => lint_map_or_none(cx, expr, arg_lists[0]),
            ["and_then", ..] => {
                let biom_option_linted = bind_instead_of_map::OptionAndThenSome::lint(cx, expr, arg_lists[0]);
                let biom_result_linted = bind_instead_of_map::ResultAndThenOk::lint(cx, expr, arg_lists[0]);
                if !biom_option_linted && !biom_result_linted {
                    unnecessary_lazy_eval::lint(cx, expr, arg_lists[0], "and");
                }
            },
            ["or_else", ..] => {
                if !bind_instead_of_map::ResultOrElseErrInfo::lint(cx, expr, arg_lists[0]) {
                    unnecessary_lazy_eval::lint(cx, expr, arg_lists[0], "or");
                }
            },
            ["next", "filter"] => lint_filter_next(cx, expr, arg_lists[1]),
            ["next", "skip_while"] => lint_skip_while_next(cx, expr, arg_lists[1]),
            ["next", "iter"] => lint_iter_next(cx, expr, arg_lists[1]),
            ["map", "filter"] => lint_filter_map(cx, expr, arg_lists[1], arg_lists[0]),
            ["map", "filter_map"] => lint_filter_map_map(cx, expr, arg_lists[1], arg_lists[0]),
            ["next", "filter_map"] => lint_filter_map_next(cx, expr, arg_lists[1]),
            ["map", "find"] => lint_find_map(cx, expr, arg_lists[1], arg_lists[0]),
            ["flat_map", "filter"] => lint_filter_flat_map(cx, expr, arg_lists[1], arg_lists[0]),
            ["flat_map", "filter_map"] => lint_filter_map_flat_map(cx, expr, arg_lists[1], arg_lists[0]),
            ["flat_map", ..] => lint_flat_map_identity(cx, expr, arg_lists[0], method_spans[0]),
            ["flatten", "map"] => lint_map_flatten(cx, expr, arg_lists[1]),
            ["is_some", "find"] => lint_search_is_some(cx, expr, "find", arg_lists[1], arg_lists[0], method_spans[1]),
            ["is_some", "position"] => {
                lint_search_is_some(cx, expr, "position", arg_lists[1], arg_lists[0], method_spans[1])
            },
            ["is_some", "rposition"] => {
                lint_search_is_some(cx, expr, "rposition", arg_lists[1], arg_lists[0], method_spans[1])
            },
            ["extend", ..] => lint_extend(cx, expr, arg_lists[0]),
            ["nth", "iter"] => lint_iter_nth(cx, expr, &arg_lists, false),
            ["nth", "iter_mut"] => lint_iter_nth(cx, expr, &arg_lists, true),
            ["nth", ..] => lint_iter_nth_zero(cx, expr, arg_lists[0]),
            ["step_by", ..] => lint_step_by(cx, expr, arg_lists[0]),
            ["next", "skip"] => lint_iter_skip_next(cx, expr, arg_lists[1]),
            ["collect", "cloned"] => lint_iter_cloned_collect(cx, expr, arg_lists[1]),
            ["as_ref"] => lint_asref(cx, expr, "as_ref", arg_lists[0]),
            ["as_mut"] => lint_asref(cx, expr, "as_mut", arg_lists[0]),
            ["fold", ..] => lint_unnecessary_fold(cx, expr, arg_lists[0], method_spans[0]),
            ["filter_map", ..] => unnecessary_filter_map::lint(cx, expr, arg_lists[0]),
            ["count", "map"] => lint_suspicious_map(cx, expr),
            ["assume_init"] => lint_maybe_uninit(cx, &arg_lists[0][0], expr),
            ["unwrap_or", arith @ ("checked_add" | "checked_sub" | "checked_mul")] => {
                manual_saturating_arithmetic::lint(cx, expr, &arg_lists, &arith["checked_".len()..])
            },
            ["add" | "offset" | "sub" | "wrapping_offset" | "wrapping_add" | "wrapping_sub"] => {
                check_pointer_offset(cx, expr, arg_lists[0])
            },
            ["is_file", ..] => lint_filetype_is_file(cx, expr, arg_lists[0]),
            ["map", "as_ref"] => {
                lint_option_as_ref_deref(cx, expr, arg_lists[1], arg_lists[0], false, self.msrv.as_ref())
            },
            ["map", "as_mut"] => {
                lint_option_as_ref_deref(cx, expr, arg_lists[1], arg_lists[0], true, self.msrv.as_ref())
            },
            ["unwrap_or_else", ..] => unnecessary_lazy_eval::lint(cx, expr, arg_lists[0], "unwrap_or"),
            ["get_or_insert_with", ..] => unnecessary_lazy_eval::lint(cx, expr, arg_lists[0], "get_or_insert"),
            ["ok_or_else", ..] => unnecessary_lazy_eval::lint(cx, expr, arg_lists[0], "ok_or"),
            ["collect", "map"] => lint_map_collect(cx, expr, arg_lists[1], arg_lists[0]),
            _ => {},
        }

        match expr.kind {
            hir::ExprKind::Call(ref func, ref args) => {
                if let hir::ExprKind::Path(path) = &func.kind {
                    if match_qpath(path, &["from_iter"]) {
                        lint_from_iter(cx, expr, args);
                    }
                }
            },
            hir::ExprKind::MethodCall(ref method_call, ref method_span, ref args, _) => {
                lint_or_fun_call(cx, expr, *method_span, &method_call.ident.as_str(), args);
                lint_expect_fun_call(cx, expr, *method_span, &method_call.ident.as_str(), args);

                let self_ty = cx.typeck_results().expr_ty_adjusted(&args[0]);
                if args.len() == 1 && method_call.ident.name == sym!(clone) {
                    lint_clone_on_copy(cx, expr, &args[0], self_ty);
                    lint_clone_on_ref_ptr(cx, expr, &args[0]);
                }
                if args.len() == 1 && method_call.ident.name == sym!(to_string) {
                    inefficient_to_string::lint(cx, expr, &args[0], self_ty);
                }

                if let Some(fn_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id) {
                    if match_def_path(cx, fn_def_id, &paths::PUSH_STR) {
                        lint_single_char_push_string(cx, expr, args);
                    } else if match_def_path(cx, fn_def_id, &paths::INSERT_STR) {
                        lint_single_char_insert_string(cx, expr, args);
                    }
                }

                match self_ty.kind() {
                    ty::Ref(_, ty, _) if *ty.kind() == ty::Str => {
                        for &(method, pos) in &PATTERN_METHODS {
                            if method_call.ident.name.as_str() == method && args.len() > pos {
                                lint_single_char_pattern(cx, expr, &args[pos]);
                            }
                        }
                    },
                    ty::Ref(..) if method_call.ident.name == sym!(into_iter) => {
                        lint_into_iter(cx, expr, self_ty, *method_span);
                    },
                    _ => (),
                }
            },
            hir::ExprKind::Binary(op, ref lhs, ref rhs)
                if op.node == hir::BinOpKind::Eq || op.node == hir::BinOpKind::Ne =>
            {
                let mut info = BinaryExprInfo {
                    expr,
                    chain: lhs,
                    other: rhs,
                    eq: op.node == hir::BinOpKind::Eq,
                };
                lint_binary_expr_with_method_call(cx, &mut info);
            }
            _ => (),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, impl_item: &'tcx hir::ImplItem<'_>) {
        if in_external_macro(cx.sess(), impl_item.span) {
            return;
        }
        let name = impl_item.ident.name.as_str();
        let parent = cx.tcx.hir().get_parent_item(impl_item.hir_id);
        let item = cx.tcx.hir().expect_item(parent);
        let def_id = cx.tcx.hir().local_def_id(item.hir_id);
        let self_ty = cx.tcx.type_of(def_id);
        if_chain! {
            if let hir::ImplItemKind::Fn(ref sig, id) = impl_item.kind;
            if let Some(first_arg) = iter_input_pats(&sig.decl, cx.tcx.hir().body(id)).next();
            if let hir::ItemKind::Impl{ of_trait: None, .. } = item.kind;

            let method_def_id = cx.tcx.hir().local_def_id(impl_item.hir_id);
            let method_sig = cx.tcx.fn_sig(method_def_id);
            let method_sig = cx.tcx.erase_late_bound_regions(method_sig);

            let first_arg_ty = &method_sig.inputs().iter().next();

            // check conventions w.r.t. conversion method names and predicates
            if let Some(first_arg_ty) = first_arg_ty;

            then {
                if cx.access_levels.is_exported(impl_item.hir_id) {
                    // check missing trait implementations
                    for method_config in &TRAIT_METHODS {
                        if name == method_config.method_name &&
                            sig.decl.inputs.len() == method_config.param_count &&
                            method_config.output_type.matches(cx, &sig.decl.output) &&
                            method_config.self_kind.matches(cx, self_ty, first_arg_ty) &&
                            fn_header_equals(method_config.fn_header, sig.header) &&
                            method_config.lifetime_param_cond(&impl_item)
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

                if let Some((ref conv, self_kinds)) = &CONVENTIONS
                    .iter()
                    .find(|(ref conv, _)| conv.check(&name))
                {
                    if !self_kinds.iter().any(|k| k.matches(cx, self_ty, first_arg_ty)) {
                        let lint = if item.vis.node.is_pub() {
                            WRONG_PUB_SELF_CONVENTION
                        } else {
                            WRONG_SELF_CONVENTION
                        };

                        span_lint(
                            cx,
                            lint,
                            first_arg.pat.span,
                            &format!("methods called `{}` usually take {}; consider choosing a less ambiguous name",
                                conv,
                                &self_kinds
                                    .iter()
                                    .map(|k| k.description())
                                    .collect::<Vec<_>>()
                                    .join(" or ")
                            ),
                        );
                    }
                }
            }
        }

        // if this impl block implements a trait, lint in trait definition instead
        if let hir::ItemKind::Impl { of_trait: Some(_), .. } = item.kind {
            return;
        }

        if let hir::ImplItemKind::Fn(_, _) = impl_item.kind {
            let ret_ty = return_ty(cx, impl_item.hir_id);

            // walk the return type and check for Self (this does not check associated types)
            if contains_ty(ret_ty, self_ty) {
                return;
            }

            // if return type is impl trait, check the associated types
            if let ty::Opaque(def_id, _) = *ret_ty.kind() {
                // one of the associated types must be Self
                for &(predicate, _span) in cx.tcx.explicit_item_bounds(def_id) {
                    if let ty::PredicateAtom::Projection(projection_predicate) = predicate.skip_binders() {
                        // walk the associated type and check for Self
                        if contains_ty(projection_predicate.ty, self_ty) {
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
        if_chain! {
            if !in_external_macro(cx.tcx.sess, item.span);
            if item.ident.name == sym::new;
            if let TraitItemKind::Fn(_, _) = item.kind;
            let ret_ty = return_ty(cx, item.hir_id);
            let self_ty = TraitRef::identity(cx.tcx, item.hir_id.owner.to_def_id()).self_ty();
            if !contains_ty(ret_ty, self_ty);

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

/// Checks for the `OR_FUN_CALL` lint.
#[allow(clippy::too_many_lines)]
fn lint_or_fun_call<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &hir::Expr<'_>,
    method_span: Span,
    name: &str,
    args: &'tcx [hir::Expr<'_>],
) {
    /// Checks for `unwrap_or(T::new())` or `unwrap_or(T::default())`.
    fn check_unwrap_or_default(
        cx: &LateContext<'_>,
        name: &str,
        fun: &hir::Expr<'_>,
        self_expr: &hir::Expr<'_>,
        arg: &hir::Expr<'_>,
        or_has_args: bool,
        span: Span,
    ) -> bool {
        if_chain! {
            if !or_has_args;
            if name == "unwrap_or";
            if let hir::ExprKind::Path(ref qpath) = fun.kind;
            let path = &*last_path_segment(qpath).ident.as_str();
            if ["default", "new"].contains(&path);
            let arg_ty = cx.typeck_results().expr_ty(arg);
            if let Some(default_trait_id) = get_trait_def_id(cx, &paths::DEFAULT_TRAIT);
            if implements_trait(cx, arg_ty, default_trait_id, &[]);

            then {
                let mut applicability = Applicability::MachineApplicable;
                span_lint_and_sugg(
                    cx,
                    OR_FUN_CALL,
                    span,
                    &format!("use of `{}` followed by a call to `{}`", name, path),
                    "try this",
                    format!(
                        "{}.unwrap_or_default()",
                        snippet_with_applicability(cx, self_expr.span, "..", &mut applicability)
                    ),
                    applicability,
                );

                true
            } else {
                false
            }
        }
    }

    /// Checks for `*or(foo())`.
    #[allow(clippy::too_many_arguments)]
    fn check_general_case<'tcx>(
        cx: &LateContext<'tcx>,
        name: &str,
        method_span: Span,
        self_expr: &hir::Expr<'_>,
        arg: &'tcx hir::Expr<'_>,
        span: Span,
        // None if lambda is required
        fun_span: Option<Span>,
    ) {
        // (path, fn_has_argument, methods, suffix)
        static KNOW_TYPES: [(&[&str], bool, &[&str], &str); 4] = [
            (&paths::BTREEMAP_ENTRY, false, &["or_insert"], "with"),
            (&paths::HASHMAP_ENTRY, false, &["or_insert"], "with"),
            (&paths::OPTION, false, &["map_or", "ok_or", "or", "unwrap_or"], "else"),
            (&paths::RESULT, true, &["or", "unwrap_or"], "else"),
        ];

        if let hir::ExprKind::MethodCall(ref path, _, ref args, _) = &arg.kind {
            if path.ident.as_str() == "len" {
                let ty = cx.typeck_results().expr_ty(&args[0]).peel_refs();

                match ty.kind() {
                    ty::Slice(_) | ty::Array(_, _) => return,
                    _ => (),
                }

                if is_type_diagnostic_item(cx, ty, sym::vec_type) {
                    return;
                }
            }
        }

        if_chain! {
            if KNOW_TYPES.iter().any(|k| k.2.contains(&name));

            if is_lazyness_candidate(cx, arg);
            if !contains_return(&arg);

            let self_ty = cx.typeck_results().expr_ty(self_expr);

            if let Some(&(_, fn_has_arguments, poss, suffix)) =
                KNOW_TYPES.iter().find(|&&i| match_type(cx, self_ty, i.0));

            if poss.contains(&name);

            then {
                let sugg: Cow<'_, str> = {
                    let (snippet_span, use_lambda) = match (fn_has_arguments, fun_span) {
                        (false, Some(fun_span)) => (fun_span, false),
                        _ => (arg.span, true),
                    };
                    let snippet = snippet_with_macro_callsite(cx, snippet_span, "..");
                    if use_lambda {
                        let l_arg = if fn_has_arguments { "_" } else { "" };
                        format!("|{}| {}", l_arg, snippet).into()
                    } else {
                        snippet
                    }
                };
                let span_replace_word = method_span.with_hi(span.hi());
                span_lint_and_sugg(
                    cx,
                    OR_FUN_CALL,
                    span_replace_word,
                    &format!("use of `{}` followed by a function call", name),
                    "try this",
                    format!("{}_{}({})", name, suffix, sugg),
                    Applicability::HasPlaceholders,
                );
            }
        }
    }

    if args.len() == 2 {
        match args[1].kind {
            hir::ExprKind::Call(ref fun, ref or_args) => {
                let or_has_args = !or_args.is_empty();
                if !check_unwrap_or_default(cx, name, fun, &args[0], &args[1], or_has_args, expr.span) {
                    let fun_span = if or_has_args { None } else { Some(fun.span) };
                    check_general_case(cx, name, method_span, &args[0], &args[1], expr.span, fun_span);
                }
            },
            hir::ExprKind::Index(..) | hir::ExprKind::MethodCall(..) => {
                check_general_case(cx, name, method_span, &args[0], &args[1], expr.span, None);
            },
            _ => {},
        }
    }
}

/// Checks for the `EXPECT_FUN_CALL` lint.
#[allow(clippy::too_many_lines)]
fn lint_expect_fun_call(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    method_span: Span,
    name: &str,
    args: &[hir::Expr<'_>],
) {
    // Strip `&`, `as_ref()` and `as_str()` off `arg` until we're left with either a `String` or
    // `&str`
    fn get_arg_root<'a>(cx: &LateContext<'_>, arg: &'a hir::Expr<'a>) -> &'a hir::Expr<'a> {
        let mut arg_root = arg;
        loop {
            arg_root = match &arg_root.kind {
                hir::ExprKind::AddrOf(hir::BorrowKind::Ref, _, expr) => expr,
                hir::ExprKind::MethodCall(method_name, _, call_args, _) => {
                    if call_args.len() == 1
                        && (method_name.ident.name == sym::as_str || method_name.ident.name == sym!(as_ref))
                        && {
                            let arg_type = cx.typeck_results().expr_ty(&call_args[0]);
                            let base_type = arg_type.peel_refs();
                            *base_type.kind() == ty::Str || is_type_diagnostic_item(cx, base_type, sym::string_type)
                        }
                    {
                        &call_args[0]
                    } else {
                        break;
                    }
                },
                _ => break,
            };
        }
        arg_root
    }

    // Only `&'static str` or `String` can be used directly in the `panic!`. Other types should be
    // converted to string.
    fn requires_to_string(cx: &LateContext<'_>, arg: &hir::Expr<'_>) -> bool {
        let arg_ty = cx.typeck_results().expr_ty(arg);
        if is_type_diagnostic_item(cx, arg_ty, sym::string_type) {
            return false;
        }
        if let ty::Ref(_, ty, ..) = arg_ty.kind() {
            if *ty.kind() == ty::Str && can_be_static_str(cx, arg) {
                return false;
            }
        };
        true
    }

    // Check if an expression could have type `&'static str`, knowing that it
    // has type `&str` for some lifetime.
    fn can_be_static_str(cx: &LateContext<'_>, arg: &hir::Expr<'_>) -> bool {
        match arg.kind {
            hir::ExprKind::Lit(_) => true,
            hir::ExprKind::Call(fun, _) => {
                if let hir::ExprKind::Path(ref p) = fun.kind {
                    match cx.qpath_res(p, fun.hir_id) {
                        hir::def::Res::Def(hir::def::DefKind::Fn | hir::def::DefKind::AssocFn, def_id) => matches!(
                            cx.tcx.fn_sig(def_id).output().skip_binder().kind(),
                            ty::Ref(ty::ReStatic, ..)
                        ),
                        _ => false,
                    }
                } else {
                    false
                }
            },
            hir::ExprKind::MethodCall(..) => {
                cx.typeck_results()
                    .type_dependent_def_id(arg.hir_id)
                    .map_or(false, |method_id| {
                        matches!(
                            cx.tcx.fn_sig(method_id).output().skip_binder().kind(),
                            ty::Ref(ty::ReStatic, ..)
                        )
                    })
            },
            hir::ExprKind::Path(ref p) => matches!(
                cx.qpath_res(p, arg.hir_id),
                hir::def::Res::Def(hir::def::DefKind::Const | hir::def::DefKind::Static, _)
            ),
            _ => false,
        }
    }

    fn generate_format_arg_snippet(
        cx: &LateContext<'_>,
        a: &hir::Expr<'_>,
        applicability: &mut Applicability,
    ) -> Vec<String> {
        if_chain! {
            if let hir::ExprKind::AddrOf(hir::BorrowKind::Ref, _, ref format_arg) = a.kind;
            if let hir::ExprKind::Match(ref format_arg_expr, _, _) = format_arg.kind;
            if let hir::ExprKind::Tup(ref format_arg_expr_tup) = format_arg_expr.kind;

            then {
                format_arg_expr_tup
                    .iter()
                    .map(|a| snippet_with_applicability(cx, a.span, "..", applicability).into_owned())
                    .collect()
            } else {
                unreachable!()
            }
        }
    }

    fn is_call(node: &hir::ExprKind<'_>) -> bool {
        match node {
            hir::ExprKind::AddrOf(hir::BorrowKind::Ref, _, expr) => {
                is_call(&expr.kind)
            },
            hir::ExprKind::Call(..)
            | hir::ExprKind::MethodCall(..)
            // These variants are debatable or require further examination
            | hir::ExprKind::Match(..)
            | hir::ExprKind::Block{ .. } => true,
            _ => false,
        }
    }

    if args.len() != 2 || name != "expect" || !is_call(&args[1].kind) {
        return;
    }

    let receiver_type = cx.typeck_results().expr_ty_adjusted(&args[0]);
    let closure_args = if is_type_diagnostic_item(cx, receiver_type, sym::option_type) {
        "||"
    } else if is_type_diagnostic_item(cx, receiver_type, sym::result_type) {
        "|_|"
    } else {
        return;
    };

    let arg_root = get_arg_root(cx, &args[1]);

    let span_replace_word = method_span.with_hi(expr.span.hi());

    let mut applicability = Applicability::MachineApplicable;

    //Special handling for `format!` as arg_root
    if_chain! {
        if let hir::ExprKind::Block(block, None) = &arg_root.kind;
        if block.stmts.len() == 1;
        if let hir::StmtKind::Local(local) = &block.stmts[0].kind;
        if let Some(arg_root) = &local.init;
        if let hir::ExprKind::Call(ref inner_fun, ref inner_args) = arg_root.kind;
        if is_expn_of(inner_fun.span, "format").is_some() && inner_args.len() == 1;
        if let hir::ExprKind::Call(_, format_args) = &inner_args[0].kind;
        then {
            let fmt_spec = &format_args[0];
            let fmt_args = &format_args[1];

            let mut args = vec![snippet(cx, fmt_spec.span, "..").into_owned()];

            args.extend(generate_format_arg_snippet(cx, fmt_args, &mut applicability));

            let sugg = args.join(", ");

            span_lint_and_sugg(
                cx,
                EXPECT_FUN_CALL,
                span_replace_word,
                &format!("use of `{}` followed by a function call", name),
                "try this",
                format!("unwrap_or_else({} panic!({}))", closure_args, sugg),
                applicability,
            );

            return;
        }
    }

    let mut arg_root_snippet: Cow<'_, _> = snippet_with_applicability(cx, arg_root.span, "..", &mut applicability);
    if requires_to_string(cx, arg_root) {
        arg_root_snippet.to_mut().push_str(".to_string()");
    }

    span_lint_and_sugg(
        cx,
        EXPECT_FUN_CALL,
        span_replace_word,
        &format!("use of `{}` followed by a function call", name),
        "try this",
        format!("unwrap_or_else({} {{ panic!({}) }})", closure_args, arg_root_snippet),
        applicability,
    );
}

/// Checks for the `CLONE_ON_COPY` lint.
fn lint_clone_on_copy(cx: &LateContext<'_>, expr: &hir::Expr<'_>, arg: &hir::Expr<'_>, arg_ty: Ty<'_>) {
    let ty = cx.typeck_results().expr_ty(expr);
    if let ty::Ref(_, inner, _) = arg_ty.kind() {
        if let ty::Ref(_, innermost, _) = inner.kind() {
            span_lint_and_then(
                cx,
                CLONE_DOUBLE_REF,
                expr.span,
                "using `clone` on a double-reference; \
                this will copy the reference instead of cloning the inner type",
                |diag| {
                    if let Some(snip) = sugg::Sugg::hir_opt(cx, arg) {
                        let mut ty = innermost;
                        let mut n = 0;
                        while let ty::Ref(_, inner, _) = ty.kind() {
                            ty = inner;
                            n += 1;
                        }
                        let refs: String = iter::repeat('&').take(n + 1).collect();
                        let derefs: String = iter::repeat('*').take(n).collect();
                        let explicit = format!("<{}{}>::clone({})", refs, ty, snip);
                        diag.span_suggestion(
                            expr.span,
                            "try dereferencing it",
                            format!("{}({}{}).clone()", refs, derefs, snip.deref()),
                            Applicability::MaybeIncorrect,
                        );
                        diag.span_suggestion(
                            expr.span,
                            "or try being explicit if you are sure, that you want to clone a reference",
                            explicit,
                            Applicability::MaybeIncorrect,
                        );
                    }
                },
            );
            return; // don't report clone_on_copy
        }
    }

    if is_copy(cx, ty) {
        let snip;
        if let Some(snippet) = sugg::Sugg::hir_opt(cx, arg) {
            let parent = cx.tcx.hir().get_parent_node(expr.hir_id);
            match &cx.tcx.hir().get(parent) {
                hir::Node::Expr(parent) => match parent.kind {
                    // &*x is a nop, &x.clone() is not
                    hir::ExprKind::AddrOf(..) => return,
                    // (*x).func() is useless, x.clone().func() can work in case func borrows mutably
                    hir::ExprKind::MethodCall(_, _, parent_args, _) if expr.hir_id == parent_args[0].hir_id => {
                        return;
                    },

                    _ => {},
                },
                hir::Node::Stmt(stmt) => {
                    if let hir::StmtKind::Local(ref loc) = stmt.kind {
                        if let hir::PatKind::Ref(..) = loc.pat.kind {
                            // let ref y = *x borrows x, let ref y = x.clone() does not
                            return;
                        }
                    }
                },
                _ => {},
            }

            // x.clone() might have dereferenced x, possibly through Deref impls
            if cx.typeck_results().expr_ty(arg) == ty {
                snip = Some(("try removing the `clone` call", format!("{}", snippet)));
            } else {
                let deref_count = cx
                    .typeck_results()
                    .expr_adjustments(arg)
                    .iter()
                    .filter(|adj| matches!(adj.kind, ty::adjustment::Adjust::Deref(_)))
                    .count();
                let derefs: String = iter::repeat('*').take(deref_count).collect();
                snip = Some(("try dereferencing it", format!("{}{}", derefs, snippet)));
            }
        } else {
            snip = None;
        }
        span_lint_and_then(cx, CLONE_ON_COPY, expr.span, "using `clone` on a `Copy` type", |diag| {
            if let Some((text, snip)) = snip {
                diag.span_suggestion(expr.span, text, snip, Applicability::MachineApplicable);
            }
        });
    }
}

fn lint_clone_on_ref_ptr(cx: &LateContext<'_>, expr: &hir::Expr<'_>, arg: &hir::Expr<'_>) {
    let obj_ty = cx.typeck_results().expr_ty(arg).peel_refs();

    if let ty::Adt(_, subst) = obj_ty.kind() {
        let caller_type = if is_type_diagnostic_item(cx, obj_ty, sym::Rc) {
            "Rc"
        } else if is_type_diagnostic_item(cx, obj_ty, sym::Arc) {
            "Arc"
        } else if match_type(cx, obj_ty, &paths::WEAK_RC) || match_type(cx, obj_ty, &paths::WEAK_ARC) {
            "Weak"
        } else {
            return;
        };

        let snippet = snippet_with_macro_callsite(cx, arg.span, "..");

        span_lint_and_sugg(
            cx,
            CLONE_ON_REF_PTR,
            expr.span,
            "using `.clone()` on a ref-counted pointer",
            "try this",
            format!("{}::<{}>::clone(&{})", caller_type, subst.type_at(0), snippet),
            Applicability::Unspecified, // Sometimes unnecessary ::<_> after Rc/Arc/Weak
        );
    }
}

fn lint_string_extend(cx: &LateContext<'_>, expr: &hir::Expr<'_>, args: &[hir::Expr<'_>]) {
    let arg = &args[1];
    if let Some(arglists) = method_chain_args(arg, &["chars"]) {
        let target = &arglists[0][0];
        let self_ty = cx.typeck_results().expr_ty(target).peel_refs();
        let ref_str = if *self_ty.kind() == ty::Str {
            ""
        } else if is_type_diagnostic_item(cx, self_ty, sym::string_type) {
            "&"
        } else {
            return;
        };

        let mut applicability = Applicability::MachineApplicable;
        span_lint_and_sugg(
            cx,
            STRING_EXTEND_CHARS,
            expr.span,
            "calling `.extend(_.chars())`",
            "try this",
            format!(
                "{}.push_str({}{})",
                snippet_with_applicability(cx, args[0].span, "..", &mut applicability),
                ref_str,
                snippet_with_applicability(cx, target.span, "..", &mut applicability)
            ),
            applicability,
        );
    }
}

fn lint_extend(cx: &LateContext<'_>, expr: &hir::Expr<'_>, args: &[hir::Expr<'_>]) {
    let obj_ty = cx.typeck_results().expr_ty(&args[0]).peel_refs();
    if is_type_diagnostic_item(cx, obj_ty, sym::string_type) {
        lint_string_extend(cx, expr, args);
    }
}

fn lint_iter_cloned_collect<'tcx>(cx: &LateContext<'tcx>, expr: &hir::Expr<'_>, iter_args: &'tcx [hir::Expr<'_>]) {
    if_chain! {
        if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(expr), sym::vec_type);
        if let Some(slice) = derefs_to_slice(cx, &iter_args[0], cx.typeck_results().expr_ty(&iter_args[0]));
        if let Some(to_replace) = expr.span.trim_start(slice.span.source_callsite());

        then {
            span_lint_and_sugg(
                cx,
                ITER_CLONED_COLLECT,
                to_replace,
                "called `iter().cloned().collect()` on a slice to create a `Vec`. Calling `to_vec()` is both faster and \
                more readable",
                "try",
                ".to_vec()".to_string(),
                Applicability::MachineApplicable,
            );
        }
    }
}

fn lint_unnecessary_fold(cx: &LateContext<'_>, expr: &hir::Expr<'_>, fold_args: &[hir::Expr<'_>], fold_span: Span) {
    fn check_fold_with_op(
        cx: &LateContext<'_>,
        expr: &hir::Expr<'_>,
        fold_args: &[hir::Expr<'_>],
        fold_span: Span,
        op: hir::BinOpKind,
        replacement_method_name: &str,
        replacement_has_args: bool,
    ) {
        if_chain! {
            // Extract the body of the closure passed to fold
            if let hir::ExprKind::Closure(_, _, body_id, _, _) = fold_args[2].kind;
            let closure_body = cx.tcx.hir().body(body_id);
            let closure_expr = remove_blocks(&closure_body.value);

            // Check if the closure body is of the form `acc <op> some_expr(x)`
            if let hir::ExprKind::Binary(ref bin_op, ref left_expr, ref right_expr) = closure_expr.kind;
            if bin_op.node == op;

            // Extract the names of the two arguments to the closure
            if let Some(first_arg_ident) = get_arg_name(&closure_body.params[0].pat);
            if let Some(second_arg_ident) = get_arg_name(&closure_body.params[1].pat);

            if match_var(&*left_expr, first_arg_ident);
            if replacement_has_args || match_var(&*right_expr, second_arg_ident);

            then {
                let mut applicability = Applicability::MachineApplicable;
                let sugg = if replacement_has_args {
                    format!(
                        "{replacement}(|{s}| {r})",
                        replacement = replacement_method_name,
                        s = second_arg_ident,
                        r = snippet_with_applicability(cx, right_expr.span, "EXPR", &mut applicability),
                    )
                } else {
                    format!(
                        "{replacement}()",
                        replacement = replacement_method_name,
                    )
                };

                span_lint_and_sugg(
                    cx,
                    UNNECESSARY_FOLD,
                    fold_span.with_hi(expr.span.hi()),
                    // TODO #2371 don't suggest e.g., .any(|x| f(x)) if we can suggest .any(f)
                    "this `.fold` can be written more succinctly using another method",
                    "try",
                    sugg,
                    applicability,
                );
            }
        }
    }

    // Check that this is a call to Iterator::fold rather than just some function called fold
    if !match_trait_method(cx, expr, &paths::ITERATOR) {
        return;
    }

    assert!(
        fold_args.len() == 3,
        "Expected fold_args to have three entries - the receiver, the initial value and the closure"
    );

    // Check if the first argument to .fold is a suitable literal
    if let hir::ExprKind::Lit(ref lit) = fold_args[1].kind {
        match lit.node {
            ast::LitKind::Bool(false) => {
                check_fold_with_op(cx, expr, fold_args, fold_span, hir::BinOpKind::Or, "any", true)
            },
            ast::LitKind::Bool(true) => {
                check_fold_with_op(cx, expr, fold_args, fold_span, hir::BinOpKind::And, "all", true)
            },
            ast::LitKind::Int(0, _) => {
                check_fold_with_op(cx, expr, fold_args, fold_span, hir::BinOpKind::Add, "sum", false)
            },
            ast::LitKind::Int(1, _) => {
                check_fold_with_op(cx, expr, fold_args, fold_span, hir::BinOpKind::Mul, "product", false)
            },
            _ => (),
        }
    }
}

fn lint_step_by<'tcx>(cx: &LateContext<'tcx>, expr: &hir::Expr<'_>, args: &'tcx [hir::Expr<'_>]) {
    if match_trait_method(cx, expr, &paths::ITERATOR) {
        if let Some((Constant::Int(0), _)) = constant(cx, cx.typeck_results(), &args[1]) {
            span_lint(
                cx,
                ITERATOR_STEP_BY_ZERO,
                expr.span,
                "Iterator::step_by(0) will panic at runtime",
            );
        }
    }
}

fn lint_iter_next<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>, iter_args: &'tcx [hir::Expr<'_>]) {
    let caller_expr = &iter_args[0];

    // Skip lint if the `iter().next()` expression is a for loop argument,
    // since it is already covered by `&loops::ITER_NEXT_LOOP`
    let mut parent_expr_opt = get_parent_expr(cx, expr);
    while let Some(parent_expr) = parent_expr_opt {
        if higher::for_loop(parent_expr).is_some() {
            return;
        }
        parent_expr_opt = get_parent_expr(cx, parent_expr);
    }

    if derefs_to_slice(cx, caller_expr, cx.typeck_results().expr_ty(caller_expr)).is_some() {
        // caller is a Slice
        if_chain! {
            if let hir::ExprKind::Index(ref caller_var, ref index_expr) = &caller_expr.kind;
            if let Some(higher::Range { start: Some(start_expr), end: None, limits: ast::RangeLimits::HalfOpen })
                = higher::range(index_expr);
            if let hir::ExprKind::Lit(ref start_lit) = &start_expr.kind;
            if let ast::LitKind::Int(start_idx, _) = start_lit.node;
            then {
                let mut applicability = Applicability::MachineApplicable;
                span_lint_and_sugg(
                    cx,
                    ITER_NEXT_SLICE,
                    expr.span,
                    "using `.iter().next()` on a Slice without end index",
                    "try calling",
                    format!("{}.get({})", snippet_with_applicability(cx, caller_var.span, "..", &mut applicability), start_idx),
                    applicability,
                );
            }
        }
    } else if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(caller_expr), sym::vec_type)
        || matches!(
            &cx.typeck_results().expr_ty(caller_expr).peel_refs().kind(),
            ty::Array(_, _)
        )
    {
        // caller is a Vec or an Array
        let mut applicability = Applicability::MachineApplicable;
        span_lint_and_sugg(
            cx,
            ITER_NEXT_SLICE,
            expr.span,
            "using `.iter().next()` on an array",
            "try calling",
            format!(
                "{}.get(0)",
                snippet_with_applicability(cx, caller_expr.span, "..", &mut applicability)
            ),
            applicability,
        );
    }
}

fn lint_iter_nth<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &hir::Expr<'_>,
    nth_and_iter_args: &[&'tcx [hir::Expr<'tcx>]],
    is_mut: bool,
) {
    let iter_args = nth_and_iter_args[1];
    let mut_str = if is_mut { "_mut" } else { "" };
    let caller_type = if derefs_to_slice(cx, &iter_args[0], cx.typeck_results().expr_ty(&iter_args[0])).is_some() {
        "slice"
    } else if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(&iter_args[0]), sym::vec_type) {
        "Vec"
    } else if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(&iter_args[0]), sym!(vecdeque_type)) {
        "VecDeque"
    } else {
        let nth_args = nth_and_iter_args[0];
        lint_iter_nth_zero(cx, expr, &nth_args);
        return; // caller is not a type that we want to lint
    };

    span_lint_and_help(
        cx,
        ITER_NTH,
        expr.span,
        &format!("called `.iter{0}().nth()` on a {1}", mut_str, caller_type),
        None,
        &format!("calling `.get{}()` is both faster and more readable", mut_str),
    );
}

fn lint_iter_nth_zero<'tcx>(cx: &LateContext<'tcx>, expr: &hir::Expr<'_>, nth_args: &'tcx [hir::Expr<'_>]) {
    if_chain! {
        if match_trait_method(cx, expr, &paths::ITERATOR);
        if let Some((Constant::Int(0), _)) = constant(cx, cx.typeck_results(), &nth_args[1]);
        then {
            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                ITER_NTH_ZERO,
                expr.span,
                "called `.nth(0)` on a `std::iter::Iterator`, when `.next()` is equivalent",
                "try calling `.next()` instead of `.nth(0)`",
                format!("{}.next()", snippet_with_applicability(cx, nth_args[0].span, "..", &mut applicability)),
                applicability,
            );
        }
    }
}

fn lint_get_unwrap<'tcx>(cx: &LateContext<'tcx>, expr: &hir::Expr<'_>, get_args: &'tcx [hir::Expr<'_>], is_mut: bool) {
    // Note: we don't want to lint `get_mut().unwrap` for `HashMap` or `BTreeMap`,
    // because they do not implement `IndexMut`
    let mut applicability = Applicability::MachineApplicable;
    let expr_ty = cx.typeck_results().expr_ty(&get_args[0]);
    let get_args_str = if get_args.len() > 1 {
        snippet_with_applicability(cx, get_args[1].span, "..", &mut applicability)
    } else {
        return; // not linting on a .get().unwrap() chain or variant
    };
    let mut needs_ref;
    let caller_type = if derefs_to_slice(cx, &get_args[0], expr_ty).is_some() {
        needs_ref = get_args_str.parse::<usize>().is_ok();
        "slice"
    } else if is_type_diagnostic_item(cx, expr_ty, sym::vec_type) {
        needs_ref = get_args_str.parse::<usize>().is_ok();
        "Vec"
    } else if is_type_diagnostic_item(cx, expr_ty, sym!(vecdeque_type)) {
        needs_ref = get_args_str.parse::<usize>().is_ok();
        "VecDeque"
    } else if !is_mut && is_type_diagnostic_item(cx, expr_ty, sym!(hashmap_type)) {
        needs_ref = true;
        "HashMap"
    } else if !is_mut && match_type(cx, expr_ty, &paths::BTREEMAP) {
        needs_ref = true;
        "BTreeMap"
    } else {
        return; // caller is not a type that we want to lint
    };

    let mut span = expr.span;

    // Handle the case where the result is immediately dereferenced
    // by not requiring ref and pulling the dereference into the
    // suggestion.
    if_chain! {
        if needs_ref;
        if let Some(parent) = get_parent_expr(cx, expr);
        if let hir::ExprKind::Unary(hir::UnOp::UnDeref, _) = parent.kind;
        then {
            needs_ref = false;
            span = parent.span;
        }
    }

    let mut_str = if is_mut { "_mut" } else { "" };
    let borrow_str = if !needs_ref {
        ""
    } else if is_mut {
        "&mut "
    } else {
        "&"
    };

    span_lint_and_sugg(
        cx,
        GET_UNWRAP,
        span,
        &format!(
            "called `.get{0}().unwrap()` on a {1}. Using `[]` is more clear and more concise",
            mut_str, caller_type
        ),
        "try this",
        format!(
            "{}{}[{}]",
            borrow_str,
            snippet_with_applicability(cx, get_args[0].span, "..", &mut applicability),
            get_args_str
        ),
        applicability,
    );
}

fn lint_iter_skip_next(cx: &LateContext<'_>, expr: &hir::Expr<'_>, skip_args: &[hir::Expr<'_>]) {
    // lint if caller of skip is an Iterator
    if match_trait_method(cx, expr, &paths::ITERATOR) {
        if let [caller, n] = skip_args {
            let hint = format!(".nth({})", snippet(cx, n.span, ".."));
            span_lint_and_sugg(
                cx,
                ITER_SKIP_NEXT,
                expr.span.trim_start(caller.span).unwrap(),
                "called `skip(..).next()` on an iterator",
                "use `nth` instead",
                hint,
                Applicability::MachineApplicable,
            );
        }
    }
}

fn derefs_to_slice<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
    ty: Ty<'tcx>,
) -> Option<&'tcx hir::Expr<'tcx>> {
    fn may_slice<'a>(cx: &LateContext<'a>, ty: Ty<'a>) -> bool {
        match ty.kind() {
            ty::Slice(_) => true,
            ty::Adt(def, _) if def.is_box() => may_slice(cx, ty.boxed_ty()),
            ty::Adt(..) => is_type_diagnostic_item(cx, ty, sym::vec_type),
            ty::Array(_, size) => size
                .try_eval_usize(cx.tcx, cx.param_env)
                .map_or(false, |size| size < 32),
            ty::Ref(_, inner, _) => may_slice(cx, inner),
            _ => false,
        }
    }

    if let hir::ExprKind::MethodCall(ref path, _, ref args, _) = expr.kind {
        if path.ident.name == sym::iter && may_slice(cx, cx.typeck_results().expr_ty(&args[0])) {
            Some(&args[0])
        } else {
            None
        }
    } else {
        match ty.kind() {
            ty::Slice(_) => Some(expr),
            ty::Adt(def, _) if def.is_box() && may_slice(cx, ty.boxed_ty()) => Some(expr),
            ty::Ref(_, inner, _) => {
                if may_slice(cx, inner) {
                    Some(expr)
                } else {
                    None
                }
            },
            _ => None,
        }
    }
}

/// lint use of `unwrap()` for `Option`s and `Result`s
fn lint_unwrap(cx: &LateContext<'_>, expr: &hir::Expr<'_>, unwrap_args: &[hir::Expr<'_>]) {
    let obj_ty = cx.typeck_results().expr_ty(&unwrap_args[0]).peel_refs();

    let mess = if is_type_diagnostic_item(cx, obj_ty, sym::option_type) {
        Some((UNWRAP_USED, "an Option", "None"))
    } else if is_type_diagnostic_item(cx, obj_ty, sym::result_type) {
        Some((UNWRAP_USED, "a Result", "Err"))
    } else {
        None
    };

    if let Some((lint, kind, none_value)) = mess {
        span_lint_and_help(
            cx,
            lint,
            expr.span,
            &format!("used `unwrap()` on `{}` value", kind,),
            None,
            &format!(
                "if you don't want to handle the `{}` case gracefully, consider \
                using `expect()` to provide a better panic message",
                none_value,
            ),
        );
    }
}

/// lint use of `expect()` for `Option`s and `Result`s
fn lint_expect(cx: &LateContext<'_>, expr: &hir::Expr<'_>, expect_args: &[hir::Expr<'_>]) {
    let obj_ty = cx.typeck_results().expr_ty(&expect_args[0]).peel_refs();

    let mess = if is_type_diagnostic_item(cx, obj_ty, sym!(option_type)) {
        Some((EXPECT_USED, "an Option", "None"))
    } else if is_type_diagnostic_item(cx, obj_ty, sym!(result_type)) {
        Some((EXPECT_USED, "a Result", "Err"))
    } else {
        None
    };

    if let Some((lint, kind, none_value)) = mess {
        span_lint_and_help(
            cx,
            lint,
            expr.span,
            &format!("used `expect()` on `{}` value", kind,),
            None,
            &format!("if this value is an `{}`, it will panic", none_value,),
        );
    }
}

/// lint use of `ok().expect()` for `Result`s
fn lint_ok_expect(cx: &LateContext<'_>, expr: &hir::Expr<'_>, ok_args: &[hir::Expr<'_>]) {
    if_chain! {
        // lint if the caller of `ok()` is a `Result`
        if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(&ok_args[0]), sym::result_type);
        let result_type = cx.typeck_results().expr_ty(&ok_args[0]);
        if let Some(error_type) = get_error_type(cx, result_type);
        if has_debug_impl(error_type, cx);

        then {
            span_lint_and_help(
                cx,
                OK_EXPECT,
                expr.span,
                "called `ok().expect()` on a `Result` value",
                None,
                "you can call `expect()` directly on the `Result`",
            );
        }
    }
}

/// lint use of `map().flatten()` for `Iterators` and 'Options'
fn lint_map_flatten<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>, map_args: &'tcx [hir::Expr<'_>]) {
    // lint if caller of `.map().flatten()` is an Iterator
    if match_trait_method(cx, expr, &paths::ITERATOR) {
        let map_closure_ty = cx.typeck_results().expr_ty(&map_args[1]);
        let is_map_to_option = match map_closure_ty.kind() {
            ty::Closure(_, _) | ty::FnDef(_, _) | ty::FnPtr(_) => {
                let map_closure_sig = match map_closure_ty.kind() {
                    ty::Closure(_, substs) => substs.as_closure().sig(),
                    _ => map_closure_ty.fn_sig(cx.tcx),
                };
                let map_closure_return_ty = cx.tcx.erase_late_bound_regions(map_closure_sig.output());
                is_type_diagnostic_item(cx, map_closure_return_ty, sym::option_type)
            },
            _ => false,
        };

        let method_to_use = if is_map_to_option {
            // `(...).map(...)` has type `impl Iterator<Item=Option<...>>
            "filter_map"
        } else {
            // `(...).map(...)` has type `impl Iterator<Item=impl Iterator<...>>
            "flat_map"
        };
        let func_snippet = snippet(cx, map_args[1].span, "..");
        let hint = format!(".{0}({1})", method_to_use, func_snippet);
        span_lint_and_sugg(
            cx,
            MAP_FLATTEN,
            expr.span.with_lo(map_args[0].span.hi()),
            "called `map(..).flatten()` on an `Iterator`",
            &format!("try using `{}` instead", method_to_use),
            hint,
            Applicability::MachineApplicable,
        );
    }

    // lint if caller of `.map().flatten()` is an Option
    if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(&map_args[0]), sym::option_type) {
        let func_snippet = snippet(cx, map_args[1].span, "..");
        let hint = format!(".and_then({})", func_snippet);
        span_lint_and_sugg(
            cx,
            MAP_FLATTEN,
            expr.span.with_lo(map_args[0].span.hi()),
            "called `map(..).flatten()` on an `Option`",
            "try using `and_then` instead",
            hint,
            Applicability::MachineApplicable,
        );
    }
}

/// lint use of `map().unwrap_or_else()` for `Option`s and `Result`s
/// Return true if lint triggered
fn lint_map_unwrap_or_else<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    map_args: &'tcx [hir::Expr<'_>],
    unwrap_args: &'tcx [hir::Expr<'_>],
) -> bool {
    // lint if the caller of `map()` is an `Option`
    let is_option = is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(&map_args[0]), sym::option_type);
    let is_result = is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(&map_args[0]), sym::result_type);

    if is_option || is_result {
        // Don't make a suggestion that may fail to compile due to mutably borrowing
        // the same variable twice.
        let map_mutated_vars = mutated_variables(&map_args[0], cx);
        let unwrap_mutated_vars = mutated_variables(&unwrap_args[1], cx);
        if let (Some(map_mutated_vars), Some(unwrap_mutated_vars)) = (map_mutated_vars, unwrap_mutated_vars) {
            if map_mutated_vars.intersection(&unwrap_mutated_vars).next().is_some() {
                return false;
            }
        } else {
            return false;
        }

        // lint message
        let msg = if is_option {
            "called `map(<f>).unwrap_or_else(<g>)` on an `Option` value. This can be done more directly by calling \
            `map_or_else(<g>, <f>)` instead"
        } else {
            "called `map(<f>).unwrap_or_else(<g>)` on a `Result` value. This can be done more directly by calling \
            `.map_or_else(<g>, <f>)` instead"
        };
        // get snippets for args to map() and unwrap_or_else()
        let map_snippet = snippet(cx, map_args[1].span, "..");
        let unwrap_snippet = snippet(cx, unwrap_args[1].span, "..");
        // lint, with note if neither arg is > 1 line and both map() and
        // unwrap_or_else() have the same span
        let multiline = map_snippet.lines().count() > 1 || unwrap_snippet.lines().count() > 1;
        let same_span = map_args[1].span.ctxt() == unwrap_args[1].span.ctxt();
        if same_span && !multiline {
            let var_snippet = snippet(cx, map_args[0].span, "..");
            span_lint_and_sugg(
                cx,
                MAP_UNWRAP_OR,
                expr.span,
                msg,
                "try this",
                format!("{}.map_or_else({}, {})", var_snippet, unwrap_snippet, map_snippet),
                Applicability::MachineApplicable,
            );
            return true;
        } else if same_span && multiline {
            span_lint(cx, MAP_UNWRAP_OR, expr.span, msg);
            return true;
        }
    }

    false
}

/// lint use of `_.map_or(None, _)` for `Option`s and `Result`s
fn lint_map_or_none<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>, map_or_args: &'tcx [hir::Expr<'_>]) {
    let is_option = is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(&map_or_args[0]), sym::option_type);
    let is_result = is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(&map_or_args[0]), sym::result_type);

    // There are two variants of this `map_or` lint:
    // (1) using `map_or` as an adapter from `Result<T,E>` to `Option<T>`
    // (2) using `map_or` as a combinator instead of `and_then`
    //
    // (For this lint) we don't care if any other type calls `map_or`
    if !is_option && !is_result {
        return;
    }

    let (lint_name, msg, instead, hint) = {
        let default_arg_is_none = if let hir::ExprKind::Path(ref qpath) = map_or_args[1].kind {
            match_qpath(qpath, &paths::OPTION_NONE)
        } else {
            return;
        };

        if !default_arg_is_none {
            // nothing to lint!
            return;
        }

        let f_arg_is_some = if let hir::ExprKind::Path(ref qpath) = map_or_args[2].kind {
            match_qpath(qpath, &paths::OPTION_SOME)
        } else {
            false
        };

        if is_option {
            let self_snippet = snippet(cx, map_or_args[0].span, "..");
            let func_snippet = snippet(cx, map_or_args[2].span, "..");
            let msg = "called `map_or(None, ..)` on an `Option` value. This can be done more directly by calling \
                       `and_then(..)` instead";
            (
                OPTION_MAP_OR_NONE,
                msg,
                "try using `and_then` instead",
                format!("{0}.and_then({1})", self_snippet, func_snippet),
            )
        } else if f_arg_is_some {
            let msg = "called `map_or(None, Some)` on a `Result` value. This can be done more directly by calling \
                       `ok()` instead";
            let self_snippet = snippet(cx, map_or_args[0].span, "..");
            (
                RESULT_MAP_OR_INTO_OPTION,
                msg,
                "try using `ok` instead",
                format!("{0}.ok()", self_snippet),
            )
        } else {
            // nothing to lint!
            return;
        }
    };

    span_lint_and_sugg(
        cx,
        lint_name,
        expr.span,
        msg,
        instead,
        hint,
        Applicability::MachineApplicable,
    );
}

/// lint use of `filter().next()` for `Iterators`
fn lint_filter_next<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>, filter_args: &'tcx [hir::Expr<'_>]) {
    // lint if caller of `.filter().next()` is an Iterator
    if match_trait_method(cx, expr, &paths::ITERATOR) {
        let msg = "called `filter(..).next()` on an `Iterator`. This is more succinctly expressed by calling \
                   `.find(..)` instead.";
        let filter_snippet = snippet(cx, filter_args[1].span, "..");
        if filter_snippet.lines().count() <= 1 {
            let iter_snippet = snippet(cx, filter_args[0].span, "..");
            // add note if not multi-line
            span_lint_and_sugg(
                cx,
                FILTER_NEXT,
                expr.span,
                msg,
                "try this",
                format!("{}.find({})", iter_snippet, filter_snippet),
                Applicability::MachineApplicable,
            );
        } else {
            span_lint(cx, FILTER_NEXT, expr.span, msg);
        }
    }
}

/// lint use of `skip_while().next()` for `Iterators`
fn lint_skip_while_next<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    _skip_while_args: &'tcx [hir::Expr<'_>],
) {
    // lint if caller of `.skip_while().next()` is an Iterator
    if match_trait_method(cx, expr, &paths::ITERATOR) {
        span_lint_and_help(
            cx,
            SKIP_WHILE_NEXT,
            expr.span,
            "called `skip_while(<p>).next()` on an `Iterator`",
            None,
            "this is more succinctly expressed by calling `.find(!<p>)` instead",
        );
    }
}

/// lint use of `filter().map()` for `Iterators`
fn lint_filter_map<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    _filter_args: &'tcx [hir::Expr<'_>],
    _map_args: &'tcx [hir::Expr<'_>],
) {
    // lint if caller of `.filter().map()` is an Iterator
    if match_trait_method(cx, expr, &paths::ITERATOR) {
        let msg = "called `filter(..).map(..)` on an `Iterator`";
        let hint = "this is more succinctly expressed by calling `.filter_map(..)` instead";
        span_lint_and_help(cx, FILTER_MAP, expr.span, msg, None, hint);
    }
}

/// lint use of `filter_map().next()` for `Iterators`
fn lint_filter_map_next<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>, filter_args: &'tcx [hir::Expr<'_>]) {
    if match_trait_method(cx, expr, &paths::ITERATOR) {
        let msg = "called `filter_map(..).next()` on an `Iterator`. This is more succinctly expressed by calling \
                   `.find_map(..)` instead.";
        let filter_snippet = snippet(cx, filter_args[1].span, "..");
        if filter_snippet.lines().count() <= 1 {
            let iter_snippet = snippet(cx, filter_args[0].span, "..");
            span_lint_and_sugg(
                cx,
                FILTER_MAP_NEXT,
                expr.span,
                msg,
                "try this",
                format!("{}.find_map({})", iter_snippet, filter_snippet),
                Applicability::MachineApplicable,
            );
        } else {
            span_lint(cx, FILTER_MAP_NEXT, expr.span, msg);
        }
    }
}

/// lint use of `find().map()` for `Iterators`
fn lint_find_map<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    _find_args: &'tcx [hir::Expr<'_>],
    map_args: &'tcx [hir::Expr<'_>],
) {
    // lint if caller of `.filter().map()` is an Iterator
    if match_trait_method(cx, &map_args[0], &paths::ITERATOR) {
        let msg = "called `find(..).map(..)` on an `Iterator`";
        let hint = "this is more succinctly expressed by calling `.find_map(..)` instead";
        span_lint_and_help(cx, FIND_MAP, expr.span, msg, None, hint);
    }
}

/// lint use of `filter_map().map()` for `Iterators`
fn lint_filter_map_map<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    _filter_args: &'tcx [hir::Expr<'_>],
    _map_args: &'tcx [hir::Expr<'_>],
) {
    // lint if caller of `.filter().map()` is an Iterator
    if match_trait_method(cx, expr, &paths::ITERATOR) {
        let msg = "called `filter_map(..).map(..)` on an `Iterator`";
        let hint = "this is more succinctly expressed by only calling `.filter_map(..)` instead";
        span_lint_and_help(cx, FILTER_MAP, expr.span, msg, None, hint);
    }
}

/// lint use of `filter().flat_map()` for `Iterators`
fn lint_filter_flat_map<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    _filter_args: &'tcx [hir::Expr<'_>],
    _map_args: &'tcx [hir::Expr<'_>],
) {
    // lint if caller of `.filter().flat_map()` is an Iterator
    if match_trait_method(cx, expr, &paths::ITERATOR) {
        let msg = "called `filter(..).flat_map(..)` on an `Iterator`";
        let hint = "this is more succinctly expressed by calling `.flat_map(..)` \
                    and filtering by returning `iter::empty()`";
        span_lint_and_help(cx, FILTER_MAP, expr.span, msg, None, hint);
    }
}

/// lint use of `filter_map().flat_map()` for `Iterators`
fn lint_filter_map_flat_map<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    _filter_args: &'tcx [hir::Expr<'_>],
    _map_args: &'tcx [hir::Expr<'_>],
) {
    // lint if caller of `.filter_map().flat_map()` is an Iterator
    if match_trait_method(cx, expr, &paths::ITERATOR) {
        let msg = "called `filter_map(..).flat_map(..)` on an `Iterator`";
        let hint = "this is more succinctly expressed by calling `.flat_map(..)` \
                    and filtering by returning `iter::empty()`";
        span_lint_and_help(cx, FILTER_MAP, expr.span, msg, None, hint);
    }
}

/// lint use of `flat_map` for `Iterators` where `flatten` would be sufficient
fn lint_flat_map_identity<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    flat_map_args: &'tcx [hir::Expr<'_>],
    flat_map_span: Span,
) {
    if match_trait_method(cx, expr, &paths::ITERATOR) {
        let arg_node = &flat_map_args[1].kind;

        let apply_lint = |message: &str| {
            span_lint_and_sugg(
                cx,
                FLAT_MAP_IDENTITY,
                flat_map_span.with_hi(expr.span.hi()),
                message,
                "try",
                "flatten()".to_string(),
                Applicability::MachineApplicable,
            );
        };

        if_chain! {
            if let hir::ExprKind::Closure(_, _, body_id, _, _) = arg_node;
            let body = cx.tcx.hir().body(*body_id);

            if let hir::PatKind::Binding(_, _, binding_ident, _) = body.params[0].pat.kind;
            if let hir::ExprKind::Path(hir::QPath::Resolved(_, ref path)) = body.value.kind;

            if path.segments.len() == 1;
            if path.segments[0].ident.as_str() == binding_ident.as_str();

            then {
                apply_lint("called `flat_map(|x| x)` on an `Iterator`");
            }
        }

        if_chain! {
            if let hir::ExprKind::Path(ref qpath) = arg_node;

            if match_qpath(qpath, &paths::STD_CONVERT_IDENTITY);

            then {
                apply_lint("called `flat_map(std::convert::identity)` on an `Iterator`");
            }
        }
    }
}

/// lint searching an Iterator followed by `is_some()`
/// or calling `find()` on a string followed by `is_some()`
fn lint_search_is_some<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    search_method: &str,
    search_args: &'tcx [hir::Expr<'_>],
    is_some_args: &'tcx [hir::Expr<'_>],
    method_span: Span,
) {
    // lint if caller of search is an Iterator
    if match_trait_method(cx, &is_some_args[0], &paths::ITERATOR) {
        let msg = format!(
            "called `is_some()` after searching an `Iterator` with `{}`",
            search_method
        );
        let hint = "this is more succinctly expressed by calling `any()`";
        let search_snippet = snippet(cx, search_args[1].span, "..");
        if search_snippet.lines().count() <= 1 {
            // suggest `any(|x| ..)` instead of `any(|&x| ..)` for `find(|&x| ..).is_some()`
            // suggest `any(|..| *..)` instead of `any(|..| **..)` for `find(|..| **..).is_some()`
            let any_search_snippet = if_chain! {
                if search_method == "find";
                if let hir::ExprKind::Closure(_, _, body_id, ..) = search_args[1].kind;
                let closure_body = cx.tcx.hir().body(body_id);
                if let Some(closure_arg) = closure_body.params.get(0);
                then {
                    if let hir::PatKind::Ref(..) = closure_arg.pat.kind {
                        Some(search_snippet.replacen('&', "", 1))
                    } else if let Some(name) = get_arg_name(&closure_arg.pat) {
                        Some(search_snippet.replace(&format!("*{}", name), &name.as_str()))
                    } else {
                        None
                    }
                } else {
                    None
                }
            };
            // add note if not multi-line
            span_lint_and_sugg(
                cx,
                SEARCH_IS_SOME,
                method_span.with_hi(expr.span.hi()),
                &msg,
                "use `any()` instead",
                format!(
                    "any({})",
                    any_search_snippet.as_ref().map_or(&*search_snippet, String::as_str)
                ),
                Applicability::MachineApplicable,
            );
        } else {
            span_lint_and_help(cx, SEARCH_IS_SOME, expr.span, &msg, None, hint);
        }
    }
    // lint if `find()` is called by `String` or `&str`
    else if search_method == "find" {
        let is_string_or_str_slice = |e| {
            let self_ty = cx.typeck_results().expr_ty(e).peel_refs();
            if is_type_diagnostic_item(cx, self_ty, sym!(string_type)) {
                true
            } else {
                *self_ty.kind() == ty::Str
            }
        };
        if_chain! {
            if is_string_or_str_slice(&search_args[0]);
            if is_string_or_str_slice(&search_args[1]);
            then {
                let msg = "called `is_some()` after calling `find()` on a string";
                let mut applicability = Applicability::MachineApplicable;
                let find_arg = snippet_with_applicability(cx, search_args[1].span, "..", &mut applicability);
                span_lint_and_sugg(
                    cx,
                    SEARCH_IS_SOME,
                    method_span.with_hi(expr.span.hi()),
                    msg,
                    "use `contains()` instead",
                    format!("contains({})", find_arg),
                    applicability,
                );
            }
        }
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
        ($func:ident, $cx:expr, $info:ident) => {
            if !$func($cx, $info) {
                ::std::mem::swap(&mut $info.chain, &mut $info.other);
                if $func($cx, $info) {
                    return;
                }
            }
        };
    }

    lint_with_both_lhs_and_rhs!(lint_chars_next_cmp, cx, info);
    lint_with_both_lhs_and_rhs!(lint_chars_last_cmp, cx, info);
    lint_with_both_lhs_and_rhs!(lint_chars_next_cmp_with_unwrap, cx, info);
    lint_with_both_lhs_and_rhs!(lint_chars_last_cmp_with_unwrap, cx, info);
}

/// Wrapper fn for `CHARS_NEXT_CMP` and `CHARS_LAST_CMP` lints.
fn lint_chars_cmp(
    cx: &LateContext<'_>,
    info: &BinaryExprInfo<'_>,
    chain_methods: &[&str],
    lint: &'static Lint,
    suggest: &str,
) -> bool {
    if_chain! {
        if let Some(args) = method_chain_args(info.chain, chain_methods);
        if let hir::ExprKind::Call(ref fun, ref arg_char) = info.other.kind;
        if arg_char.len() == 1;
        if let hir::ExprKind::Path(ref qpath) = fun.kind;
        if let Some(segment) = single_segment_path(qpath);
        if segment.ident.name == sym::Some;
        then {
            let mut applicability = Applicability::MachineApplicable;
            let self_ty = cx.typeck_results().expr_ty_adjusted(&args[0][0]).peel_refs();

            if *self_ty.kind() != ty::Str {
                return false;
            }

            span_lint_and_sugg(
                cx,
                lint,
                info.expr.span,
                &format!("you should use the `{}` method", suggest),
                "like this",
                format!("{}{}.{}({})",
                        if info.eq { "" } else { "!" },
                        snippet_with_applicability(cx, args[0][0].span, "..", &mut applicability),
                        suggest,
                        snippet_with_applicability(cx, arg_char[0].span, "..", &mut applicability)),
                applicability,
            );

            return true;
        }
    }

    false
}

/// Checks for the `CHARS_NEXT_CMP` lint.
fn lint_chars_next_cmp<'tcx>(cx: &LateContext<'tcx>, info: &BinaryExprInfo<'_>) -> bool {
    lint_chars_cmp(cx, info, &["chars", "next"], CHARS_NEXT_CMP, "starts_with")
}

/// Checks for the `CHARS_LAST_CMP` lint.
fn lint_chars_last_cmp<'tcx>(cx: &LateContext<'tcx>, info: &BinaryExprInfo<'_>) -> bool {
    if lint_chars_cmp(cx, info, &["chars", "last"], CHARS_LAST_CMP, "ends_with") {
        true
    } else {
        lint_chars_cmp(cx, info, &["chars", "next_back"], CHARS_LAST_CMP, "ends_with")
    }
}

/// Wrapper fn for `CHARS_NEXT_CMP` and `CHARS_LAST_CMP` lints with `unwrap()`.
fn lint_chars_cmp_with_unwrap<'tcx>(
    cx: &LateContext<'tcx>,
    info: &BinaryExprInfo<'_>,
    chain_methods: &[&str],
    lint: &'static Lint,
    suggest: &str,
) -> bool {
    if_chain! {
        if let Some(args) = method_chain_args(info.chain, chain_methods);
        if let hir::ExprKind::Lit(ref lit) = info.other.kind;
        if let ast::LitKind::Char(c) = lit.node;
        then {
            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                lint,
                info.expr.span,
                &format!("you should use the `{}` method", suggest),
                "like this",
                format!("{}{}.{}('{}')",
                        if info.eq { "" } else { "!" },
                        snippet_with_applicability(cx, args[0][0].span, "..", &mut applicability),
                        suggest,
                        c),
                applicability,
            );

            true
        } else {
            false
        }
    }
}

/// Checks for the `CHARS_NEXT_CMP` lint with `unwrap()`.
fn lint_chars_next_cmp_with_unwrap<'tcx>(cx: &LateContext<'tcx>, info: &BinaryExprInfo<'_>) -> bool {
    lint_chars_cmp_with_unwrap(cx, info, &["chars", "next", "unwrap"], CHARS_NEXT_CMP, "starts_with")
}

/// Checks for the `CHARS_LAST_CMP` lint with `unwrap()`.
fn lint_chars_last_cmp_with_unwrap<'tcx>(cx: &LateContext<'tcx>, info: &BinaryExprInfo<'_>) -> bool {
    if lint_chars_cmp_with_unwrap(cx, info, &["chars", "last", "unwrap"], CHARS_LAST_CMP, "ends_with") {
        true
    } else {
        lint_chars_cmp_with_unwrap(cx, info, &["chars", "next_back", "unwrap"], CHARS_LAST_CMP, "ends_with")
    }
}

fn get_hint_if_single_char_arg(
    cx: &LateContext<'_>,
    arg: &hir::Expr<'_>,
    applicability: &mut Applicability,
) -> Option<String> {
    if_chain! {
        if let hir::ExprKind::Lit(lit) = &arg.kind;
        if let ast::LitKind::Str(r, style) = lit.node;
        let string = r.as_str();
        if string.chars().count() == 1;
        then {
            let snip = snippet_with_applicability(cx, arg.span, &string, applicability);
            let ch = if let ast::StrStyle::Raw(nhash) = style {
                let nhash = nhash as usize;
                // for raw string: r##"a"##
                &snip[(nhash + 2)..(snip.len() - 1 - nhash)]
            } else {
                // for regular string: "a"
                &snip[1..(snip.len() - 1)]
            };
            let hint = format!("'{}'", if ch == "'" { "\\'" } else { ch });
            Some(hint)
        } else {
            None
        }
    }
}

/// lint for length-1 `str`s for methods in `PATTERN_METHODS`
fn lint_single_char_pattern(cx: &LateContext<'_>, _expr: &hir::Expr<'_>, arg: &hir::Expr<'_>) {
    let mut applicability = Applicability::MachineApplicable;
    if let Some(hint) = get_hint_if_single_char_arg(cx, arg, &mut applicability) {
        span_lint_and_sugg(
            cx,
            SINGLE_CHAR_PATTERN,
            arg.span,
            "single-character string constant used as pattern",
            "try using a `char` instead",
            hint,
            applicability,
        );
    }
}

/// lint for length-1 `str`s as argument for `push_str`
fn lint_single_char_push_string(cx: &LateContext<'_>, expr: &hir::Expr<'_>, args: &[hir::Expr<'_>]) {
    let mut applicability = Applicability::MachineApplicable;
    if let Some(extension_string) = get_hint_if_single_char_arg(cx, &args[1], &mut applicability) {
        let base_string_snippet =
            snippet_with_applicability(cx, args[0].span.source_callsite(), "..", &mut applicability);
        let sugg = format!("{}.push({})", base_string_snippet, extension_string);
        span_lint_and_sugg(
            cx,
            SINGLE_CHAR_ADD_STR,
            expr.span,
            "calling `push_str()` using a single-character string literal",
            "consider using `push` with a character literal",
            sugg,
            applicability,
        );
    }
}

/// lint for length-1 `str`s as argument for `insert_str`
fn lint_single_char_insert_string(cx: &LateContext<'_>, expr: &hir::Expr<'_>, args: &[hir::Expr<'_>]) {
    let mut applicability = Applicability::MachineApplicable;
    if let Some(extension_string) = get_hint_if_single_char_arg(cx, &args[2], &mut applicability) {
        let base_string_snippet =
            snippet_with_applicability(cx, args[0].span.source_callsite(), "_", &mut applicability);
        let pos_arg = snippet_with_applicability(cx, args[1].span, "..", &mut applicability);
        let sugg = format!("{}.insert({}, {})", base_string_snippet, pos_arg, extension_string);
        span_lint_and_sugg(
            cx,
            SINGLE_CHAR_ADD_STR,
            expr.span,
            "calling `insert_str()` using a single-character string literal",
            "consider using `insert` with a character literal",
            sugg,
            applicability,
        );
    }
}

/// Checks for the `USELESS_ASREF` lint.
fn lint_asref(cx: &LateContext<'_>, expr: &hir::Expr<'_>, call_name: &str, as_ref_args: &[hir::Expr<'_>]) {
    // when we get here, we've already checked that the call name is "as_ref" or "as_mut"
    // check if the call is to the actual `AsRef` or `AsMut` trait
    if match_trait_method(cx, expr, &paths::ASREF_TRAIT) || match_trait_method(cx, expr, &paths::ASMUT_TRAIT) {
        // check if the type after `as_ref` or `as_mut` is the same as before
        let recvr = &as_ref_args[0];
        let rcv_ty = cx.typeck_results().expr_ty(recvr);
        let res_ty = cx.typeck_results().expr_ty(expr);
        let (base_res_ty, res_depth) = walk_ptrs_ty_depth(res_ty);
        let (base_rcv_ty, rcv_depth) = walk_ptrs_ty_depth(rcv_ty);
        if base_rcv_ty == base_res_ty && rcv_depth >= res_depth {
            // allow the `as_ref` or `as_mut` if it is followed by another method call
            if_chain! {
                if let Some(parent) = get_parent_expr(cx, expr);
                if let hir::ExprKind::MethodCall(_, ref span, _, _) = parent.kind;
                if span != &expr.span;
                then {
                    return;
                }
            }

            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                USELESS_ASREF,
                expr.span,
                &format!("this call to `{}` does nothing", call_name),
                "try this",
                snippet_with_applicability(cx, recvr.span, "..", &mut applicability).to_string(),
                applicability,
            );
        }
    }
}

fn ty_has_iter_method(cx: &LateContext<'_>, self_ref_ty: Ty<'_>) -> Option<(&'static str, &'static str)> {
    has_iter_method(cx, self_ref_ty).map(|ty_name| {
        let mutbl = match self_ref_ty.kind() {
            ty::Ref(_, _, mutbl) => mutbl,
            _ => unreachable!(),
        };
        let method_name = match mutbl {
            hir::Mutability::Not => "iter",
            hir::Mutability::Mut => "iter_mut",
        };
        (ty_name, method_name)
    })
}

fn lint_into_iter(cx: &LateContext<'_>, expr: &hir::Expr<'_>, self_ref_ty: Ty<'_>, method_span: Span) {
    if !match_trait_method(cx, expr, &paths::INTO_ITERATOR) {
        return;
    }
    if let Some((kind, method_name)) = ty_has_iter_method(cx, self_ref_ty) {
        span_lint_and_sugg(
            cx,
            INTO_ITER_ON_REF,
            method_span,
            &format!(
                "this `.into_iter()` call is equivalent to `.{}()` and will not consume the `{}`",
                method_name, kind,
            ),
            "call directly",
            method_name.to_string(),
            Applicability::MachineApplicable,
        );
    }
}

/// lint for `MaybeUninit::uninit().assume_init()` (we already have the latter)
fn lint_maybe_uninit(cx: &LateContext<'_>, expr: &hir::Expr<'_>, outer: &hir::Expr<'_>) {
    if_chain! {
        if let hir::ExprKind::Call(ref callee, ref args) = expr.kind;
        if args.is_empty();
        if let hir::ExprKind::Path(ref path) = callee.kind;
        if match_qpath(path, &paths::MEM_MAYBEUNINIT_UNINIT);
        if !is_maybe_uninit_ty_valid(cx, cx.typeck_results().expr_ty_adjusted(outer));
        then {
            span_lint(
                cx,
                UNINIT_ASSUMED_INIT,
                outer.span,
                "this call for this type may be undefined behavior"
            );
        }
    }
}

fn is_maybe_uninit_ty_valid(cx: &LateContext<'_>, ty: Ty<'_>) -> bool {
    match ty.kind() {
        ty::Array(ref component, _) => is_maybe_uninit_ty_valid(cx, component),
        ty::Tuple(ref types) => types.types().all(|ty| is_maybe_uninit_ty_valid(cx, ty)),
        ty::Adt(ref adt, _) => match_def_path(cx, adt.did, &paths::MEM_MAYBEUNINIT),
        _ => false,
    }
}

fn lint_suspicious_map(cx: &LateContext<'_>, expr: &hir::Expr<'_>) {
    span_lint_and_help(
        cx,
        SUSPICIOUS_MAP,
        expr.span,
        "this call to `map()` won't have an effect on the call to `count()`",
        None,
        "make sure you did not confuse `map` with `filter` or `for_each`",
    );
}

const OPTION_AS_REF_DEREF_MSRV: RustcVersion = RustcVersion::new(1, 40, 0);

/// lint use of `_.as_ref().map(Deref::deref)` for `Option`s
fn lint_option_as_ref_deref<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &hir::Expr<'_>,
    as_ref_args: &[hir::Expr<'_>],
    map_args: &[hir::Expr<'_>],
    is_mut: bool,
    msrv: Option<&RustcVersion>,
) {
    if !meets_msrv(msrv, &OPTION_AS_REF_DEREF_MSRV) {
        return;
    }

    let same_mutability = |m| (is_mut && m == &hir::Mutability::Mut) || (!is_mut && m == &hir::Mutability::Not);

    let option_ty = cx.typeck_results().expr_ty(&as_ref_args[0]);
    if !is_type_diagnostic_item(cx, option_ty, sym::option_type) {
        return;
    }

    let deref_aliases: [&[&str]; 9] = [
        &paths::DEREF_TRAIT_METHOD,
        &paths::DEREF_MUT_TRAIT_METHOD,
        &paths::CSTRING_AS_C_STR,
        &paths::OS_STRING_AS_OS_STR,
        &paths::PATH_BUF_AS_PATH,
        &paths::STRING_AS_STR,
        &paths::STRING_AS_MUT_STR,
        &paths::VEC_AS_SLICE,
        &paths::VEC_AS_MUT_SLICE,
    ];

    let is_deref = match map_args[1].kind {
        hir::ExprKind::Path(ref expr_qpath) => cx
            .qpath_res(expr_qpath, map_args[1].hir_id)
            .opt_def_id()
            .map_or(false, |fun_def_id| {
                deref_aliases.iter().any(|path| match_def_path(cx, fun_def_id, path))
            }),
        hir::ExprKind::Closure(_, _, body_id, _, _) => {
            let closure_body = cx.tcx.hir().body(body_id);
            let closure_expr = remove_blocks(&closure_body.value);

            match &closure_expr.kind {
                hir::ExprKind::MethodCall(_, _, args, _) => {
                    if_chain! {
                        if args.len() == 1;
                        if let hir::ExprKind::Path(qpath) = &args[0].kind;
                        if let hir::def::Res::Local(local_id) = cx.qpath_res(qpath, args[0].hir_id);
                        if closure_body.params[0].pat.hir_id == local_id;
                        let adj = cx
                            .typeck_results()
                            .expr_adjustments(&args[0])
                            .iter()
                            .map(|x| &x.kind)
                            .collect::<Box<[_]>>();
                        if let [ty::adjustment::Adjust::Deref(None), ty::adjustment::Adjust::Borrow(_)] = *adj;
                        then {
                            let method_did = cx.typeck_results().type_dependent_def_id(closure_expr.hir_id).unwrap();
                            deref_aliases.iter().any(|path| match_def_path(cx, method_did, path))
                        } else {
                            false
                        }
                    }
                },
                hir::ExprKind::AddrOf(hir::BorrowKind::Ref, m, ref inner) if same_mutability(m) => {
                    if_chain! {
                        if let hir::ExprKind::Unary(hir::UnOp::UnDeref, ref inner1) = inner.kind;
                        if let hir::ExprKind::Unary(hir::UnOp::UnDeref, ref inner2) = inner1.kind;
                        if let hir::ExprKind::Path(ref qpath) = inner2.kind;
                        if let hir::def::Res::Local(local_id) = cx.qpath_res(qpath, inner2.hir_id);
                        then {
                            closure_body.params[0].pat.hir_id == local_id
                        } else {
                            false
                        }
                    }
                },
                _ => false,
            }
        },
        _ => false,
    };

    if is_deref {
        let current_method = if is_mut {
            format!(".as_mut().map({})", snippet(cx, map_args[1].span, ".."))
        } else {
            format!(".as_ref().map({})", snippet(cx, map_args[1].span, ".."))
        };
        let method_hint = if is_mut { "as_deref_mut" } else { "as_deref" };
        let hint = format!("{}.{}()", snippet(cx, as_ref_args[0].span, ".."), method_hint);
        let suggestion = format!("try using {} instead", method_hint);

        let msg = format!(
            "called `{0}` on an Option value. This can be done more directly \
            by calling `{1}` instead",
            current_method, hint
        );
        span_lint_and_sugg(
            cx,
            OPTION_AS_REF_DEREF,
            expr.span,
            &msg,
            &suggestion,
            hint,
            Applicability::MachineApplicable,
        );
    }
}

fn lint_map_collect(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    map_args: &[hir::Expr<'_>],
    collect_args: &[hir::Expr<'_>],
) {
    if_chain! {
        // called on Iterator
        if let [map_expr] = collect_args;
        if match_trait_method(cx, map_expr, &paths::ITERATOR);
        // return of collect `Result<(),_>`
        let collect_ret_ty = cx.typeck_results().expr_ty(expr);
        if is_type_diagnostic_item(cx, collect_ret_ty, sym::result_type);
        if let ty::Adt(_, substs) = collect_ret_ty.kind();
        if let Some(result_t) = substs.types().next();
        if result_t.is_unit();
        // get parts for snippet
        if let [iter, map_fn] = map_args;
        then {
            span_lint_and_sugg(
                cx,
                MAP_COLLECT_RESULT_UNIT,
                expr.span,
                "`.map().collect()` can be replaced with `.try_for_each()`",
                "try this",
                format!(
                    "{}.try_for_each({})",
                    snippet(cx, iter.span, ".."),
                    snippet(cx, map_fn.span, "..")
                ),
                Applicability::MachineApplicable,
            );
        }
    }
}

/// Given a `Result<T, E>` type, return its error type (`E`).
fn get_error_type<'a>(cx: &LateContext<'_>, ty: Ty<'a>) -> Option<Ty<'a>> {
    match ty.kind() {
        ty::Adt(_, substs) if is_type_diagnostic_item(cx, ty, sym::result_type) => substs.types().nth(1),
        _ => None,
    }
}

/// This checks whether a given type is known to implement Debug.
fn has_debug_impl<'tcx>(ty: Ty<'tcx>, cx: &LateContext<'tcx>) -> bool {
    cx.tcx
        .get_diagnostic_item(sym::debug_trait)
        .map_or(false, |debug| implements_trait(cx, ty, debug, &[]))
}

enum Convention {
    Eq(&'static str),
    StartsWith(&'static str),
}

#[rustfmt::skip]
const CONVENTIONS: [(Convention, &[SelfKind]); 7] = [
    (Convention::Eq("new"), &[SelfKind::No]),
    (Convention::StartsWith("as_"), &[SelfKind::Ref, SelfKind::RefMut]),
    (Convention::StartsWith("from_"), &[SelfKind::No]),
    (Convention::StartsWith("into_"), &[SelfKind::Value]),
    (Convention::StartsWith("is_"), &[SelfKind::Ref, SelfKind::No]),
    (Convention::Eq("to_mut"), &[SelfKind::RefMut]),
    (Convention::StartsWith("to_"), &[SelfKind::Ref]),
];

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

#[rustfmt::skip]
const PATTERN_METHODS: [(&str, usize); 17] = [
    ("contains", 1),
    ("starts_with", 1),
    ("ends_with", 1),
    ("find", 1),
    ("rfind", 1),
    ("split", 1),
    ("rsplit", 1),
    ("split_terminator", 1),
    ("rsplit_terminator", 1),
    ("splitn", 2),
    ("rsplitn", 2),
    ("matches", 1),
    ("rmatches", 1),
    ("match_indices", 1),
    ("rmatch_indices", 1),
    ("trim_start_matches", 1),
    ("trim_end_matches", 1),
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
            Self::Value => "self by value",
            Self::Ref => "self by reference",
            Self::RefMut => "self by mutable reference",
            Self::No => "no self",
        }
    }
}

impl Convention {
    #[must_use]
    fn check(&self, other: &str) -> bool {
        match *self {
            Self::Eq(this) => this == other,
            Self::StartsWith(this) => other.starts_with(this) && this != other,
        }
    }
}

impl fmt::Display for Convention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match *self {
            Self::Eq(this) => this.fmt(f),
            Self::StartsWith(this) => this.fmt(f).and_then(|_| '*'.fmt(f)),
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
    fn matches(self, cx: &LateContext<'_>, ty: &hir::FnRetTy<'_>) -> bool {
        let is_unit = |ty: &hir::Ty<'_>| SpanlessEq::new(cx).eq_ty_kind(&ty.kind, &hir::TyKind::Tup(&[]));
        match (self, ty) {
            (Self::Unit, &hir::FnRetTy::DefaultReturn(_)) => true,
            (Self::Unit, &hir::FnRetTy::Return(ref ty)) if is_unit(ty) => true,
            (Self::Bool, &hir::FnRetTy::Return(ref ty)) if is_bool(ty) => true,
            (Self::Any, &hir::FnRetTy::Return(ref ty)) if !is_unit(ty) => true,
            (Self::Ref, &hir::FnRetTy::Return(ref ty)) => matches!(ty.kind, hir::TyKind::Rptr(_, _)),
            _ => false,
        }
    }
}

fn is_bool(ty: &hir::Ty<'_>) -> bool {
    if let hir::TyKind::Path(ref p) = ty.kind {
        match_qpath(p, &["bool"])
    } else {
        false
    }
}

fn check_pointer_offset(cx: &LateContext<'_>, expr: &hir::Expr<'_>, args: &[hir::Expr<'_>]) {
    if_chain! {
        if args.len() == 2;
        if let ty::RawPtr(ty::TypeAndMut { ref ty, .. }) = cx.typeck_results().expr_ty(&args[0]).kind();
        if let Ok(layout) = cx.tcx.layout_of(cx.param_env.and(ty));
        if layout.is_zst();
        then {
            span_lint(cx, ZST_OFFSET, expr.span, "offset calculation on zero-sized value");
        }
    }
}

fn lint_filetype_is_file(cx: &LateContext<'_>, expr: &hir::Expr<'_>, args: &[hir::Expr<'_>]) {
    let ty = cx.typeck_results().expr_ty(&args[0]);

    if !match_type(cx, ty, &paths::FILE_TYPE) {
        return;
    }

    let span: Span;
    let verb: &str;
    let lint_unary: &str;
    let help_unary: &str;
    if_chain! {
        if let Some(parent) = get_parent_expr(cx, expr);
        if let hir::ExprKind::Unary(op, _) = parent.kind;
        if op == hir::UnOp::UnNot;
        then {
            lint_unary = "!";
            verb = "denies";
            help_unary = "";
            span = parent.span;
        } else {
            lint_unary = "";
            verb = "covers";
            help_unary = "!";
            span = expr.span;
        }
    }
    let lint_msg = format!("`{}FileType::is_file()` only {} regular files", lint_unary, verb);
    let help_msg = format!("use `{}FileType::is_dir()` instead", help_unary);
    span_lint_and_help(cx, FILETYPE_IS_FILE, span, &lint_msg, None, &help_msg);
}

fn lint_from_iter(cx: &LateContext<'_>, expr: &hir::Expr<'_>, args: &[hir::Expr<'_>]) {
    let ty = cx.typeck_results().expr_ty(expr);
    let arg_ty = cx.typeck_results().expr_ty(&args[0]);

    if_chain! {
        if let Some(from_iter_id) = get_trait_def_id(cx, &paths::FROM_ITERATOR);
        if let Some(iter_id) = get_trait_def_id(cx, &paths::ITERATOR);

        if implements_trait(cx, ty, from_iter_id, &[]) && implements_trait(cx, arg_ty, iter_id, &[]);
        then {
            // `expr` implements `FromIterator` trait
            let iter_expr = snippet(cx, args[0].span, "..");
            span_lint_and_sugg(
                cx,
                FROM_ITER_INSTEAD_OF_COLLECT,
                expr.span,
                "usage of `FromIterator::from_iter`",
                "use `.collect()` instead of `::from_iter()`",
                format!("{}.collect()", iter_expr),
                Applicability::MaybeIncorrect,
            );
        }
    }
}

fn fn_header_equals(expected: hir::FnHeader, actual: hir::FnHeader) -> bool {
    expected.constness == actual.constness
        && expected.unsafety == actual.unsafety
        && expected.asyncness == actual.asyncness
}
