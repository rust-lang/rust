//! Some lints that are built in to the compiler.
//!
//! These are the built-in lints that are emitted direct in the main
//! compiler code, rather than using their own custom pass. Those
//! lints are all available in `rustc_lint::builtin`.

use crate::{declare_lint, declare_lint_pass, declare_tool_lint};
use rustc_span::edition::Edition;
use rustc_span::symbol::sym;

declare_lint! {
    /// The `ill_formed_attribute_input` lint detects ill-formed attribute
    /// inputs that were previously accepted and used in practice.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #[inline = "this is not valid"]
    /// fn foo() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Previously, inputs for many built-in attributes weren't validated and
    /// nonsensical attribute inputs were accepted. After validation was
    /// added, it was determined that some existing projects made use of these
    /// invalid forms. This is a [future-incompatible] lint to transition this
    /// to a hard error in the future. See [issue #57571] for more details.
    ///
    /// Check the [attribute reference] for details on the valid inputs for
    /// attributes.
    ///
    /// [issue #57571]: https://github.com/rust-lang/rust/issues/57571
    /// [attribute reference]: https://doc.rust-lang.org/nightly/reference/attributes.html
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub ILL_FORMED_ATTRIBUTE_INPUT,
    Deny,
    "ill-formed attribute inputs that were previously accepted and used in practice",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #57571 <https://github.com/rust-lang/rust/issues/57571>",
        edition: None,
    };
    crate_level_only
}

declare_lint! {
    /// The `conflicting_repr_hints` lint detects [`repr` attributes] with
    /// conflicting hints.
    ///
    /// [`repr` attributes]: https://doc.rust-lang.org/reference/type-layout.html#representations
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #[repr(u32, u64)]
    /// enum Foo {
    ///     Variant1,
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The compiler incorrectly accepted these conflicting representations in
    /// the past. This is a [future-incompatible] lint to transition this to a
    /// hard error in the future. See [issue #68585] for more details.
    ///
    /// To correct the issue, remove one of the conflicting hints.
    ///
    /// [issue #68585]: https://github.com/rust-lang/rust/issues/68585
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub CONFLICTING_REPR_HINTS,
    Deny,
    "conflicts between `#[repr(..)]` hints that were previously accepted and used in practice",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #68585 <https://github.com/rust-lang/rust/issues/68585>",
        edition: None,
    };
}

declare_lint! {
    /// The `meta_variable_misuse` lint detects possible meta-variable misuse
    /// in macro definitions.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(meta_variable_misuse)]
    ///
    /// macro_rules! foo {
    ///     () => {};
    ///     ($( $i:ident = $($j:ident),+ );*) => { $( $( $i = $k; )+ )* };
    /// }
    ///
    /// fn main() {
    ///     foo!();
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// There are quite a few different ways a [`macro_rules`] macro can be
    /// improperly defined. Many of these errors were previously only detected
    /// when the macro was expanded or not at all. This lint is an attempt to
    /// catch some of these problems when the macro is *defined*.
    ///
    /// This lint is "allow" by default because it may have false positives
    /// and other issues. See [issue #61053] for more details.
    ///
    /// [`macro_rules`]: https://doc.rust-lang.org/reference/macros-by-example.html
    /// [issue #61053]: https://github.com/rust-lang/rust/issues/61053
    pub META_VARIABLE_MISUSE,
    Allow,
    "possible meta-variable misuse at macro definition"
}

declare_lint! {
    /// The `incomplete_include` lint detects the use of the [`include!`]
    /// macro with a file that contains more than one expression.
    ///
    /// [`include!`]: https://doc.rust-lang.org/std/macro.include.html
    ///
    /// ### Example
    ///
    /// ```rust,ignore (needs separate file)
    /// fn main() {
    ///     include!("foo.txt");
    /// }
    /// ```
    ///
    /// where the file `foo.txt` contains:
    ///
    /// ```text
    /// println!("hi!");
    /// ```
    ///
    /// produces:
    ///
    /// ```text
    /// error: include macro expected single expression in source
    ///  --> foo.txt:1:14
    ///   |
    /// 1 | println!("1");
    ///   |              ^
    ///   |
    ///   = note: `#[deny(incomplete_include)]` on by default
    /// ```
    ///
    /// ### Explanation
    ///
    /// The [`include!`] macro is currently only intended to be used to
    /// include a single [expression] or multiple [items]. Historically it
    /// would ignore any contents after the first expression, but that can be
    /// confusing. In the example above, the `println!` expression ends just
    /// before the semicolon, making the semicolon "extra" information that is
    /// ignored. Perhaps even more surprising, if the included file had
    /// multiple print statements, the subsequent ones would be ignored!
    ///
    /// One workaround is to place the contents in braces to create a [block
    /// expression]. Also consider alternatives, like using functions to
    /// encapsulate the expressions, or use [proc-macros].
    ///
    /// This is a lint instead of a hard error because existing projects were
    /// found to hit this error. To be cautious, it is a lint for now. The
    /// future semantics of the `include!` macro are also uncertain, see
    /// [issue #35560].
    ///
    /// [items]: https://doc.rust-lang.org/reference/items.html
    /// [expression]: https://doc.rust-lang.org/reference/expressions.html
    /// [block expression]: https://doc.rust-lang.org/reference/expressions/block-expr.html
    /// [proc-macros]: https://doc.rust-lang.org/reference/procedural-macros.html
    /// [issue #35560]: https://github.com/rust-lang/rust/issues/35560
    pub INCOMPLETE_INCLUDE,
    Deny,
    "trailing content in included file"
}

declare_lint! {
    /// The `arithmetic_overflow` lint detects that an arithmetic operation
    /// will [overflow].
    ///
    /// [overflow]: https://doc.rust-lang.org/reference/expressions/operator-expr.html#overflow
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// 1_i32 << 32;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// It is very likely a mistake to perform an arithmetic operation that
    /// overflows its value. If the compiler is able to detect these kinds of
    /// overflows at compile-time, it will trigger this lint. Consider
    /// adjusting the expression to avoid overflow, or use a data type that
    /// will not overflow.
    pub ARITHMETIC_OVERFLOW,
    Deny,
    "arithmetic operation overflows"
}

declare_lint! {
    /// The `unconditional_panic` lint detects an operation that will cause a
    /// panic at runtime.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// # #![allow(unused)]
    /// let x = 1 / 0;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// This lint detects code that is very likely incorrect because it will
    /// always panic, such as division by zero and out-of-bounds array
    /// accesses. Consider adjusting your code if this is a bug, or using the
    /// `panic!` or `unreachable!` macro instead in case the panic is intended.
    pub UNCONDITIONAL_PANIC,
    Deny,
    "operation will cause a panic at runtime"
}

declare_lint! {
    /// The `const_err` lint detects an erroneous expression while doing
    /// constant evaluation.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![allow(unconditional_panic)]
    /// let x: &'static i32 = &(1 / 0);
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// This lint detects code that is very likely incorrect. If this lint is
    /// allowed, then the code will not be evaluated at compile-time, and
    /// instead continue to generate code to evaluate at runtime, which may
    /// panic during runtime.
    ///
    /// Note that this lint may trigger in either inside or outside of a
    /// [const context]. Outside of a [const context], the compiler can
    /// sometimes evaluate an expression at compile-time in order to generate
    /// more efficient code. As the compiler becomes better at doing this, it
    /// needs to decide what to do when it encounters code that it knows for
    /// certain will panic or is otherwise incorrect. Making this a hard error
    /// would prevent existing code that exhibited this behavior from
    /// compiling, breaking backwards-compatibility. However, this is almost
    /// certainly incorrect code, so this is a deny-by-default lint. For more
    /// details, see [RFC 1229] and [issue #28238].
    ///
    /// Note that there are several other more specific lints associated with
    /// compile-time evaluation, such as [`arithmetic_overflow`],
    /// [`unconditional_panic`].
    ///
    /// [const context]: https://doc.rust-lang.org/reference/const_eval.html#const-context
    /// [RFC 1229]: https://github.com/rust-lang/rfcs/blob/master/text/1229-compile-time-asserts.md
    /// [issue #28238]: https://github.com/rust-lang/rust/issues/28238
    /// [`arithmetic_overflow`]: deny-by-default.html#arithmetic-overflow
    /// [`unconditional_panic`]: deny-by-default.html#unconditional-panic
    pub CONST_ERR,
    Deny,
    "constant evaluation detected erroneous expression",
    report_in_external_macro
}

declare_lint! {
    /// The `unused_imports` lint detects imports that are never used.
    ///
    /// ### Example
    ///
    /// ```rust
    /// use std::collections::HashMap;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Unused imports may signal a mistake or unfinished code, and clutter
    /// the code, and should be removed. If you intended to re-export the item
    /// to make it available outside of the module, add a visibility modifier
    /// like `pub`.
    pub UNUSED_IMPORTS,
    Warn,
    "imports that are never used"
}

declare_lint! {
    /// The `unused_extern_crates` lint guards against `extern crate` items
    /// that are never used.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(unused_extern_crates)]
    /// extern crate proc_macro;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// `extern crate` items that are unused have no effect and should be
    /// removed. Note that there are some cases where specifying an `extern
    /// crate` is desired for the side effect of ensuring the given crate is
    /// linked, even though it is not otherwise directly referenced. The lint
    /// can be silenced by aliasing the crate to an underscore, such as
    /// `extern crate foo as _`. Also note that it is no longer idiomatic to
    /// use `extern crate` in the [2018 edition], as extern crates are now
    /// automatically added in scope.
    ///
    /// This lint is "allow" by default because it can be noisy, and produce
    /// false-positives. If a dependency is being removed from a project, it
    /// is recommended to remove it from the build configuration (such as
    /// `Cargo.toml`) to ensure stale build entries aren't left behind.
    ///
    /// [2018 edition]: https://doc.rust-lang.org/edition-guide/rust-2018/module-system/path-clarity.html#no-more-extern-crate
    pub UNUSED_EXTERN_CRATES,
    Allow,
    "extern crates that are never used"
}

declare_lint! {
    /// The `unused_crate_dependencies` lint detects crate dependencies that
    /// are never used.
    ///
    /// ### Example
    ///
    /// ```rust,ignore (needs extern crate)
    /// #![deny(unused_crate_dependencies)]
    /// ```
    ///
    /// This will produce:
    ///
    /// ```text
    /// error: external crate `regex` unused in `lint_example`: remove the dependency or add `use regex as _;`
    ///   |
    /// note: the lint level is defined here
    ///  --> src/lib.rs:1:9
    ///   |
    /// 1 | #![deny(unused_crate_dependencies)]
    ///   |         ^^^^^^^^^^^^^^^^^^^^^^^^^
    /// ```
    ///
    /// ### Explanation
    ///
    /// After removing the code that uses a dependency, this usually also
    /// requires removing the dependency from the build configuration.
    /// However, sometimes that step can be missed, which leads to time wasted
    /// building dependencies that are no longer used. This lint can be
    /// enabled to detect dependencies that are never used (more specifically,
    /// any dependency passed with the `--extern` command-line flag that is
    /// never referenced via [`use`], [`extern crate`], or in any [path]).
    ///
    /// This lint is "allow" by default because it can provide false positives
    /// depending on how the build system is configured. For example, when
    /// using Cargo, a "package" consists of multiple crates (such as a
    /// library and a binary), but the dependencies are defined for the
    /// package as a whole. If there is a dependency that is only used in the
    /// binary, but not the library, then the lint will be incorrectly issued
    /// in the library.
    ///
    /// [path]: https://doc.rust-lang.org/reference/paths.html
    /// [`use`]: https://doc.rust-lang.org/reference/items/use-declarations.html
    /// [`extern crate`]: https://doc.rust-lang.org/reference/items/extern-crates.html
    pub UNUSED_CRATE_DEPENDENCIES,
    Allow,
    "crate dependencies that are never used",
    crate_level_only
}

declare_lint! {
    /// The `unused_qualifications` lint detects unnecessarily qualified
    /// names.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(unused_qualifications)]
    /// mod foo {
    ///     pub fn bar() {}
    /// }
    ///
    /// fn main() {
    ///     use foo::bar;
    ///     foo::bar();
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// If an item from another module is already brought into scope, then
    /// there is no need to qualify it in this case. You can call `bar()`
    /// directly, without the `foo::`.
    ///
    /// This lint is "allow" by default because it is somewhat pedantic, and
    /// doesn't indicate an actual problem, but rather a stylistic choice, and
    /// can be noisy when refactoring or moving around code.
    pub UNUSED_QUALIFICATIONS,
    Allow,
    "detects unnecessarily qualified names"
}

declare_lint! {
    /// The `unknown_lints` lint detects unrecognized lint attribute.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![allow(not_a_real_lint)]
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// It is usually a mistake to specify a lint that does not exist. Check
    /// the spelling, and check the lint listing for the correct name. Also
    /// consider if you are using an old version of the compiler, and the lint
    /// is only available in a newer version.
    pub UNKNOWN_LINTS,
    Warn,
    "unrecognized lint attribute"
}

declare_lint! {
    /// The `unused_variables` lint detects variables which are not used in
    /// any way.
    ///
    /// ### Example
    ///
    /// ```rust
    /// let x = 5;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Unused variables may signal a mistake or unfinished code. To silence
    /// the warning for the individual variable, prefix it with an underscore
    /// such as `_x`.
    pub UNUSED_VARIABLES,
    Warn,
    "detect variables which are not used in any way"
}

declare_lint! {
    /// The `unused_assignments` lint detects assignments that will never be read.
    ///
    /// ### Example
    ///
    /// ```rust
    /// let mut x = 5;
    /// x = 6;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Unused assignments may signal a mistake or unfinished code. If the
    /// variable is never used after being assigned, then the assignment can
    /// be removed. Variables with an underscore prefix such as `_x` will not
    /// trigger this lint.
    pub UNUSED_ASSIGNMENTS,
    Warn,
    "detect assignments that will never be read"
}

declare_lint! {
    /// The `dead_code` lint detects unused, unexported items.
    ///
    /// ### Example
    ///
    /// ```rust
    /// fn foo() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Dead code may signal a mistake or unfinished code. To silence the
    /// warning for individual items, prefix the name with an underscore such
    /// as `_foo`. If it was intended to expose the item outside of the crate,
    /// consider adding a visibility modifier like `pub`. Otherwise consider
    /// removing the unused code.
    pub DEAD_CODE,
    Warn,
    "detect unused, unexported items"
}

declare_lint! {
    /// The `unused_attributes` lint detects attributes that were not used by
    /// the compiler.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![ignore]
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Unused [attributes] may indicate the attribute is placed in the wrong
    /// position. Consider removing it, or placing it in the correct position.
    /// Also consider if you intended to use an _inner attribute_ (with a `!`
    /// such as `#![allow(unused)]`) which applies to the item the attribute
    /// is within, or an _outer attribute_ (without a `!` such as
    /// `#[allow(unsued)]`) which applies to the item *following* the
    /// attribute.
    ///
    /// [attributes]: https://doc.rust-lang.org/reference/attributes.html
    pub UNUSED_ATTRIBUTES,
    Warn,
    "detects attributes that were not used by the compiler"
}

declare_lint! {
    /// The `unreachable_code` lint detects unreachable code paths.
    ///
    /// ### Example
    ///
    /// ```rust,no_run
    /// panic!("we never go past here!");
    ///
    /// let x = 5;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Unreachable code may signal a mistake or unfinished code. If the code
    /// is no longer in use, consider removing it.
    pub UNREACHABLE_CODE,
    Warn,
    "detects unreachable code paths",
    report_in_external_macro
}

declare_lint! {
    /// The `unreachable_patterns` lint detects unreachable patterns.
    ///
    /// ### Example
    ///
    /// ```rust
    /// let x = 5;
    /// match x {
    ///     y => (),
    ///     5 => (),
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// This usually indicates a mistake in how the patterns are specified or
    /// ordered. In this example, the `y` pattern will always match, so the
    /// five is impossible to reach. Remember, match arms match in order, you
    /// probably wanted to put the `5` case above the `y` case.
    pub UNREACHABLE_PATTERNS,
    Warn,
    "detects unreachable patterns"
}

declare_lint! {
    /// The `overlapping_range_endpoints` lint detects `match` arms that have [range patterns] that
    /// overlap on their endpoints.
    ///
    /// [range patterns]: https://doc.rust-lang.org/nightly/reference/patterns.html#range-patterns
    ///
    /// ### Example
    ///
    /// ```rust
    /// let x = 123u8;
    /// match x {
    ///     0..=100 => { println!("small"); }
    ///     100..=255 => { println!("large"); }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// It is likely a mistake to have range patterns in a match expression that overlap in this
    /// way. Check that the beginning and end values are what you expect, and keep in mind that
    /// with `..=` the left and right bounds are inclusive.
    pub OVERLAPPING_RANGE_ENDPOINTS,
    Warn,
    "detects range patterns with overlapping endpoints"
}

declare_lint! {
    /// The `bindings_with_variant_name` lint detects pattern bindings with
    /// the same name as one of the matched variants.
    ///
    /// ### Example
    ///
    /// ```rust
    /// pub enum Enum {
    ///     Foo,
    ///     Bar,
    /// }
    ///
    /// pub fn foo(x: Enum) {
    ///     match x {
    ///         Foo => {}
    ///         Bar => {}
    ///     }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// It is usually a mistake to specify an enum variant name as an
    /// [identifier pattern]. In the example above, the `match` arms are
    /// specifying a variable name to bind the value of `x` to. The second arm
    /// is ignored because the first one matches *all* values. The likely
    /// intent is that the arm was intended to match on the enum variant.
    ///
    /// Two possible solutions are:
    ///
    /// * Specify the enum variant using a [path pattern], such as
    ///   `Enum::Foo`.
    /// * Bring the enum variants into local scope, such as adding `use
    ///   Enum::*;` to the beginning of the `foo` function in the example
    ///   above.
    ///
    /// [identifier pattern]: https://doc.rust-lang.org/reference/patterns.html#identifier-patterns
    /// [path pattern]: https://doc.rust-lang.org/reference/patterns.html#path-patterns
    pub BINDINGS_WITH_VARIANT_NAME,
    Warn,
    "detects pattern bindings with the same name as one of the matched variants"
}

declare_lint! {
    /// The `unused_macros` lint detects macros that were not used.
    ///
    /// ### Example
    ///
    /// ```rust
    /// macro_rules! unused {
    ///     () => {};
    /// }
    ///
    /// fn main() {
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Unused macros may signal a mistake or unfinished code. To silence the
    /// warning for the individual macro, prefix the name with an underscore
    /// such as `_my_macro`. If you intended to export the macro to make it
    /// available outside of the crate, use the [`macro_export` attribute].
    ///
    /// [`macro_export` attribute]: https://doc.rust-lang.org/reference/macros-by-example.html#path-based-scope
    pub UNUSED_MACROS,
    Warn,
    "detects macros that were not used"
}

declare_lint! {
    /// The `warnings` lint allows you to change the level of other
    /// lints which produce warnings.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![deny(warnings)]
    /// fn foo() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The `warnings` lint is a bit special; by changing its level, you
    /// change every other warning that would produce a warning to whatever
    /// value you'd like. As such, you won't ever trigger this lint in your
    /// code directly.
    pub WARNINGS,
    Warn,
    "mass-change the level for lints which produce warnings"
}

declare_lint! {
    /// The `unused_features` lint detects unused or unknown features found in
    /// crate-level [`feature` attributes].
    ///
    /// [`feature` attributes]: https://doc.rust-lang.org/nightly/unstable-book/
    ///
    /// Note: This lint is currently not functional, see [issue #44232] for
    /// more details.
    ///
    /// [issue #44232]: https://github.com/rust-lang/rust/issues/44232
    pub UNUSED_FEATURES,
    Warn,
    "unused features found in crate-level `#[feature]` directives"
}

declare_lint! {
    /// The `stable_features` lint detects a [`feature` attribute] that
    /// has since been made stable.
    ///
    /// [`feature` attribute]: https://doc.rust-lang.org/nightly/unstable-book/
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![feature(test_accepted_feature)]
    /// fn main() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// When a feature is stabilized, it is no longer necessary to include a
    /// `#![feature]` attribute for it. To fix, simply remove the
    /// `#![feature]` attribute.
    pub STABLE_FEATURES,
    Warn,
    "stable features found in `#[feature]` directive"
}

declare_lint! {
    /// The `unknown_crate_types` lint detects an unknown crate type found in
    /// a [`crate_type` attribute].
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![crate_type="lol"]
    /// fn main() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// An unknown value give to the `crate_type` attribute is almost
    /// certainly a mistake.
    ///
    /// [`crate_type` attribute]: https://doc.rust-lang.org/reference/linkage.html
    pub UNKNOWN_CRATE_TYPES,
    Deny,
    "unknown crate type found in `#[crate_type]` directive",
    crate_level_only
}

declare_lint! {
    /// The `trivial_casts` lint detects trivial casts which could be replaced
    /// with coercion, which may require [type ascription] or a temporary
    /// variable.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(trivial_casts)]
    /// let x: &u32 = &42;
    /// let y = x as *const u32;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// A trivial cast is a cast `e as T` where `e` has type `U` and `U` is a
    /// subtype of `T`. This type of cast is usually unnecessary, as it can be
    /// usually be inferred.
    ///
    /// This lint is "allow" by default because there are situations, such as
    /// with FFI interfaces or complex type aliases, where it triggers
    /// incorrectly, or in situations where it will be more difficult to
    /// clearly express the intent. It may be possible that this will become a
    /// warning in the future, possibly with [type ascription] providing a
    /// convenient way to work around the current issues. See [RFC 401] for
    /// historical context.
    ///
    /// [type ascription]: https://github.com/rust-lang/rust/issues/23416
    /// [RFC 401]: https://github.com/rust-lang/rfcs/blob/master/text/0401-coercions.md
    pub TRIVIAL_CASTS,
    Allow,
    "detects trivial casts which could be removed"
}

declare_lint! {
    /// The `trivial_numeric_casts` lint detects trivial numeric casts of types
    /// which could be removed.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(trivial_numeric_casts)]
    /// let x = 42_i32 as i32;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// A trivial numeric cast is a cast of a numeric type to the same numeric
    /// type. This type of cast is usually unnecessary.
    ///
    /// This lint is "allow" by default because there are situations, such as
    /// with FFI interfaces or complex type aliases, where it triggers
    /// incorrectly, or in situations where it will be more difficult to
    /// clearly express the intent. It may be possible that this will become a
    /// warning in the future, possibly with [type ascription] providing a
    /// convenient way to work around the current issues. See [RFC 401] for
    /// historical context.
    ///
    /// [type ascription]: https://github.com/rust-lang/rust/issues/23416
    /// [RFC 401]: https://github.com/rust-lang/rfcs/blob/master/text/0401-coercions.md
    pub TRIVIAL_NUMERIC_CASTS,
    Allow,
    "detects trivial casts of numeric types which could be removed"
}

declare_lint! {
    /// The `private_in_public` lint detects private items in public
    /// interfaces not caught by the old implementation.
    ///
    /// ### Example
    ///
    /// ```rust
    /// # #![allow(unused)]
    /// struct SemiPriv;
    ///
    /// mod m1 {
    ///     struct Priv;
    ///     impl super::SemiPriv {
    ///         pub fn f(_: Priv) {}
    ///     }
    /// }
    /// # fn main() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The visibility rules are intended to prevent exposing private items in
    /// public interfaces. This is a [future-incompatible] lint to transition
    /// this to a hard error in the future. See [issue #34537] for more
    /// details.
    ///
    /// [issue #34537]: https://github.com/rust-lang/rust/issues/34537
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub PRIVATE_IN_PUBLIC,
    Warn,
    "detect private items in public interfaces not caught by the old implementation",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #34537 <https://github.com/rust-lang/rust/issues/34537>",
        edition: None,
    };
}

declare_lint! {
    /// The `exported_private_dependencies` lint detects private dependencies
    /// that are exposed in a public interface.
    ///
    /// ### Example
    ///
    /// ```rust,ignore (needs-dependency)
    /// pub fn foo() -> Option<some_private_dependency::Thing> {
    ///     None
    /// }
    /// ```
    ///
    /// This will produce:
    ///
    /// ```text
    /// warning: type `bar::Thing` from private dependency 'bar' in public interface
    ///  --> src/lib.rs:3:1
    ///   |
    /// 3 | pub fn foo() -> Option<bar::Thing> {
    ///   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ///   |
    ///   = note: `#[warn(exported_private_dependencies)]` on by default
    /// ```
    ///
    /// ### Explanation
    ///
    /// Dependencies can be marked as "private" to indicate that they are not
    /// exposed in the public interface of a crate. This can be used by Cargo
    /// to independently resolve those dependencies because it can assume it
    /// does not need to unify them with other packages using that same
    /// dependency. This lint is an indication of a violation of that
    /// contract.
    ///
    /// To fix this, avoid exposing the dependency in your public interface.
    /// Or, switch the dependency to a public dependency.
    ///
    /// Note that support for this is only available on the nightly channel.
    /// See [RFC 1977] for more details, as well as the [Cargo documentation].
    ///
    /// [RFC 1977]: https://github.com/rust-lang/rfcs/blob/master/text/1977-public-private-dependencies.md
    /// [Cargo documentation]: https://doc.rust-lang.org/nightly/cargo/reference/unstable.html#public-dependency
    pub EXPORTED_PRIVATE_DEPENDENCIES,
    Warn,
    "public interface leaks type from a private dependency"
}

declare_lint! {
    /// The `pub_use_of_private_extern_crate` lint detects a specific
    /// situation of re-exporting a private `extern crate`.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// extern crate core;
    /// pub use core as reexported_core;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// A public `use` declaration should not be used to publicly re-export a
    /// private `extern crate`. `pub extern crate` should be used instead.
    ///
    /// This was historically allowed, but is not the intended behavior
    /// according to the visibility rules. This is a [future-incompatible]
    /// lint to transition this to a hard error in the future. See [issue
    /// #34537] for more details.
    ///
    /// [issue #34537]: https://github.com/rust-lang/rust/issues/34537
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub PUB_USE_OF_PRIVATE_EXTERN_CRATE,
    Deny,
    "detect public re-exports of private extern crates",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #34537 <https://github.com/rust-lang/rust/issues/34537>",
        edition: None,
    };
}

declare_lint! {
    /// The `invalid_type_param_default` lint detects type parameter defaults
    /// erroneously allowed in an invalid location.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// fn foo<T=i32>(t: T) {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Default type parameters were only intended to be allowed in certain
    /// situations, but historically the compiler allowed them everywhere.
    /// This is a [future-incompatible] lint to transition this to a hard
    /// error in the future. See [issue #36887] for more details.
    ///
    /// [issue #36887]: https://github.com/rust-lang/rust/issues/36887
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub INVALID_TYPE_PARAM_DEFAULT,
    Deny,
    "type parameter default erroneously allowed in invalid location",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #36887 <https://github.com/rust-lang/rust/issues/36887>",
        edition: None,
    };
}

declare_lint! {
    /// The `renamed_and_removed_lints` lint detects lints that have been
    /// renamed or removed.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![deny(raw_pointer_derive)]
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// To fix this, either remove the lint or use the new name. This can help
    /// avoid confusion about lints that are no longer valid, and help
    /// maintain consistency for renamed lints.
    pub RENAMED_AND_REMOVED_LINTS,
    Warn,
    "lints that have been renamed or removed"
}

declare_lint! {
    /// The `unaligned_references` lint detects unaligned references to fields
    /// of [packed] structs.
    ///
    /// [packed]: https://doc.rust-lang.org/reference/type-layout.html#the-alignment-modifiers
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(unaligned_references)]
    ///
    /// #[repr(packed)]
    /// pub struct Foo {
    ///     field1: u64,
    ///     field2: u8,
    /// }
    ///
    /// fn main() {
    ///     unsafe {
    ///         let foo = Foo { field1: 0, field2: 0 };
    ///         let _ = &foo.field1;
    ///     }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Creating a reference to an insufficiently aligned packed field is
    /// [undefined behavior] and should be disallowed.
    ///
    /// This lint is "allow" by default because there is no stable
    /// alternative, and it is not yet certain how widespread existing code
    /// will trigger this lint.
    ///
    /// See [issue #27060] for more discussion.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    /// [issue #27060]: https://github.com/rust-lang/rust/issues/27060
    pub UNALIGNED_REFERENCES,
    Allow,
    "detects unaligned references to fields of packed structs",
}

declare_lint! {
    /// The `const_item_mutation` lint detects attempts to mutate a `const`
    /// item.
    ///
    /// ### Example
    ///
    /// ```rust
    /// const FOO: [i32; 1] = [0];
    ///
    /// fn main() {
    ///     FOO[0] = 1;
    ///     // This will print "[0]".
    ///     println!("{:?}", FOO);
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Trying to directly mutate a `const` item is almost always a mistake.
    /// What is happening in the example above is that a temporary copy of the
    /// `const` is mutated, but the original `const` is not. Each time you
    /// refer to the `const` by name (such as `FOO` in the example above), a
    /// separate copy of the value is inlined at that location.
    ///
    /// This lint checks for writing directly to a field (`FOO.field =
    /// some_value`) or array entry (`FOO[0] = val`), or taking a mutable
    /// reference to the const item (`&mut FOO`), including through an
    /// autoderef (`FOO.some_mut_self_method()`).
    ///
    /// There are various alternatives depending on what you are trying to
    /// accomplish:
    ///
    /// * First, always reconsider using mutable globals, as they can be
    ///   difficult to use correctly, and can make the code more difficult to
    ///   use or understand.
    /// * If you are trying to perform a one-time initialization of a global:
    ///     * If the value can be computed at compile-time, consider using
    ///       const-compatible values (see [Constant Evaluation]).
    ///     * For more complex single-initialization cases, consider using a
    ///       third-party crate, such as [`lazy_static`] or [`once_cell`].
    ///     * If you are using the [nightly channel], consider the new
    ///       [`lazy`] module in the standard library.
    /// * If you truly need a mutable global, consider using a [`static`],
    ///   which has a variety of options:
    ///   * Simple data types can be directly defined and mutated with an
    ///     [`atomic`] type.
    ///   * More complex types can be placed in a synchronization primitive
    ///     like a [`Mutex`], which can be initialized with one of the options
    ///     listed above.
    ///   * A [mutable `static`] is a low-level primitive, requiring unsafe.
    ///     Typically This should be avoided in preference of something
    ///     higher-level like one of the above.
    ///
    /// [Constant Evaluation]: https://doc.rust-lang.org/reference/const_eval.html
    /// [`static`]: https://doc.rust-lang.org/reference/items/static-items.html
    /// [mutable `static`]: https://doc.rust-lang.org/reference/items/static-items.html#mutable-statics
    /// [`lazy`]: https://doc.rust-lang.org/nightly/std/lazy/index.html
    /// [`lazy_static`]: https://crates.io/crates/lazy_static
    /// [`once_cell`]: https://crates.io/crates/once_cell
    /// [`atomic`]: https://doc.rust-lang.org/std/sync/atomic/index.html
    /// [`Mutex`]: https://doc.rust-lang.org/std/sync/struct.Mutex.html
    pub CONST_ITEM_MUTATION,
    Warn,
    "detects attempts to mutate a `const` item",
}

declare_lint! {
    /// The `safe_packed_borrows` lint detects borrowing a field in the
    /// interior of a packed structure with alignment other than 1.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #[repr(packed)]
    /// pub struct Unaligned<T>(pub T);
    ///
    /// pub struct Foo {
    ///     start: u8,
    ///     data: Unaligned<u32>,
    /// }
    ///
    /// fn main() {
    ///     let x = Foo { start: 0, data: Unaligned(1) };
    ///     let y = &x.data.0;
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// This type of borrow is unsafe and can cause errors on some platforms
    /// and violates some assumptions made by the compiler. This was
    /// previously allowed unintentionally. This is a [future-incompatible]
    /// lint to transition this to a hard error in the future. See [issue
    /// #46043] for more details, including guidance on how to solve the
    /// problem.
    ///
    /// [issue #46043]: https://github.com/rust-lang/rust/issues/46043
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub SAFE_PACKED_BORROWS,
    Warn,
    "safe borrows of fields of packed structs were erroneously allowed",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #46043 <https://github.com/rust-lang/rust/issues/46043>",
        edition: None,
    };
}

declare_lint! {
    /// The `patterns_in_fns_without_body` lint detects `mut` identifier
    /// patterns as a parameter in functions without a body.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// trait Trait {
    ///     fn foo(mut arg: u8);
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// To fix this, remove `mut` from the parameter in the trait definition;
    /// it can be used in the implementation. That is, the following is OK:
    ///
    /// ```rust
    /// trait Trait {
    ///     fn foo(arg: u8); // Removed `mut` here
    /// }
    ///
    /// impl Trait for i32 {
    ///     fn foo(mut arg: u8) { // `mut` here is OK
    ///
    ///     }
    /// }
    /// ```
    ///
    /// Trait definitions can define functions without a body to specify a
    /// function that implementors must define. The parameter names in the
    /// body-less functions are only allowed to be `_` or an [identifier] for
    /// documentation purposes (only the type is relevant). Previous versions
    /// of the compiler erroneously allowed [identifier patterns] with the
    /// `mut` keyword, but this was not intended to be allowed. This is a
    /// [future-incompatible] lint to transition this to a hard error in the
    /// future. See [issue #35203] for more details.
    ///
    /// [identifier]: https://doc.rust-lang.org/reference/identifiers.html
    /// [identifier patterns]: https://doc.rust-lang.org/reference/patterns.html#identifier-patterns
    /// [issue #35203]: https://github.com/rust-lang/rust/issues/35203
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub PATTERNS_IN_FNS_WITHOUT_BODY,
    Deny,
    "patterns in functions without body were erroneously allowed",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #35203 <https://github.com/rust-lang/rust/issues/35203>",
        edition: None,
    };
}

declare_lint! {
    /// The `missing_fragment_specifier` lint is issued when an unused pattern in a
    /// `macro_rules!` macro definition has a meta-variable (e.g. `$e`) that is not
    /// followed by a fragment specifier (e.g. `:expr`).
    ///
    /// This warning can always be fixed by removing the unused pattern in the
    /// `macro_rules!` macro definition.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// macro_rules! foo {
    ///    () => {};
    ///    ($name) => { };
    /// }
    ///
    /// fn main() {
    ///    foo!();
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// To fix this, remove the unused pattern from the `macro_rules!` macro definition:
    ///
    /// ```rust
    /// macro_rules! foo {
    ///     () => {};
    /// }
    /// fn main() {
    ///     foo!();
    /// }
    /// ```
    pub MISSING_FRAGMENT_SPECIFIER,
    Deny,
    "detects missing fragment specifiers in unused `macro_rules!` patterns",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #40107 <https://github.com/rust-lang/rust/issues/40107>",
        edition: None,
    };
}

declare_lint! {
    /// The `late_bound_lifetime_arguments` lint detects generic lifetime
    /// arguments in path segments with late bound lifetime parameters.
    ///
    /// ### Example
    ///
    /// ```rust
    /// struct S;
    ///
    /// impl S {
    ///     fn late<'a, 'b>(self, _: &'a u8, _: &'b u8) {}
    /// }
    ///
    /// fn main() {
    ///     S.late::<'static>(&0, &0);
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// It is not clear how to provide arguments for early-bound lifetime
    /// parameters if they are intermixed with late-bound parameters in the
    /// same list. For now, providing any explicit arguments will trigger this
    /// lint if late-bound parameters are present, so in the future a solution
    /// can be adopted without hitting backward compatibility issues. This is
    /// a [future-incompatible] lint to transition this to a hard error in the
    /// future. See [issue #42868] for more details, along with a description
    /// of the difference between early and late-bound parameters.
    ///
    /// [issue #42868]: https://github.com/rust-lang/rust/issues/42868
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub LATE_BOUND_LIFETIME_ARGUMENTS,
    Warn,
    "detects generic lifetime arguments in path segments with late bound lifetime parameters",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #42868 <https://github.com/rust-lang/rust/issues/42868>",
        edition: None,
    };
}

declare_lint! {
    /// The `order_dependent_trait_objects` lint detects a trait coherency
    /// violation that would allow creating two trait impls for the same
    /// dynamic trait object involving marker traits.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// pub trait Trait {}
    ///
    /// impl Trait for dyn Send + Sync { }
    /// impl Trait for dyn Sync + Send { }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// A previous bug caused the compiler to interpret traits with different
    /// orders (such as `Send + Sync` and `Sync + Send`) as distinct types
    /// when they were intended to be treated the same. This allowed code to
    /// define separate trait implementations when there should be a coherence
    /// error. This is a [future-incompatible] lint to transition this to a
    /// hard error in the future. See [issue #56484] for more details.
    ///
    /// [issue #56484]: https://github.com/rust-lang/rust/issues/56484
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub ORDER_DEPENDENT_TRAIT_OBJECTS,
    Deny,
    "trait-object types were treated as different depending on marker-trait order",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #56484 <https://github.com/rust-lang/rust/issues/56484>",
        edition: None,
    };
}

declare_lint! {
    /// The `coherence_leak_check` lint detects conflicting implementations of
    /// a trait that are only distinguished by the old leak-check code.
    ///
    /// ### Example
    ///
    /// ```rust
    /// trait SomeTrait { }
    /// impl SomeTrait for for<'a> fn(&'a u8) { }
    /// impl<'a> SomeTrait for fn(&'a u8) { }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// In the past, the compiler would accept trait implementations for
    /// identical functions that differed only in where the lifetime binder
    /// appeared. Due to a change in the borrow checker implementation to fix
    /// several bugs, this is no longer allowed. However, since this affects
    /// existing code, this is a [future-incompatible] lint to transition this
    /// to a hard error in the future.
    ///
    /// Code relying on this pattern should introduce "[newtypes]",
    /// like `struct Foo(for<'a> fn(&'a u8))`.
    ///
    /// See [issue #56105] for more details.
    ///
    /// [issue #56105]: https://github.com/rust-lang/rust/issues/56105
    /// [newtypes]: https://doc.rust-lang.org/book/ch19-04-advanced-types.html#using-the-newtype-pattern-for-type-safety-and-abstraction
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub COHERENCE_LEAK_CHECK,
    Warn,
    "distinct impls distinguished only by the leak-check code",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #56105 <https://github.com/rust-lang/rust/issues/56105>",
        edition: None,
    };
}

declare_lint! {
    /// The `deprecated` lint detects use of deprecated items.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #[deprecated]
    /// fn foo() {}
    ///
    /// fn bar() {
    ///     foo();
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Items may be marked "deprecated" with the [`deprecated` attribute] to
    /// indicate that they should no longer be used. Usually the attribute
    /// should include a note on what to use instead, or check the
    /// documentation.
    ///
    /// [`deprecated` attribute]: https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-deprecated-attribute
    pub DEPRECATED,
    Warn,
    "detects use of deprecated items",
    report_in_external_macro
}

declare_lint! {
    /// The `unused_unsafe` lint detects unnecessary use of an `unsafe` block.
    ///
    /// ### Example
    ///
    /// ```rust
    /// unsafe {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// If nothing within the block requires `unsafe`, then remove the
    /// `unsafe` marker because it is not required and may cause confusion.
    pub UNUSED_UNSAFE,
    Warn,
    "unnecessary use of an `unsafe` block"
}

declare_lint! {
    /// The `unused_mut` lint detects mut variables which don't need to be
    /// mutable.
    ///
    /// ### Example
    ///
    /// ```rust
    /// let mut x = 5;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The preferred style is to only mark variables as `mut` if it is
    /// required.
    pub UNUSED_MUT,
    Warn,
    "detect mut variables which don't need to be mutable"
}

declare_lint! {
    /// The `unconditional_recursion` lint detects functions that cannot
    /// return without calling themselves.
    ///
    /// ### Example
    ///
    /// ```rust
    /// fn foo() {
    ///     foo();
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// It is usually a mistake to have a recursive call that does not have
    /// some condition to cause it to terminate. If you really intend to have
    /// an infinite loop, using a `loop` expression is recommended.
    pub UNCONDITIONAL_RECURSION,
    Warn,
    "functions that cannot return without calling themselves"
}

declare_lint! {
    /// The `single_use_lifetimes` lint detects lifetimes that are only used
    /// once.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(single_use_lifetimes)]
    ///
    /// fn foo<'a>(x: &'a u32) {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Specifying an explicit lifetime like `'a` in a function or `impl`
    /// should only be used to link together two things. Otherwise, you should
    /// just use `'_` to indicate that the lifetime is not linked to anything,
    /// or elide the lifetime altogether if possible.
    ///
    /// This lint is "allow" by default because it was introduced at a time
    /// when `'_` and elided lifetimes were first being introduced, and this
    /// lint would be too noisy. Also, there are some known false positives
    /// that it produces. See [RFC 2115] for historical context, and [issue
    /// #44752] for more details.
    ///
    /// [RFC 2115]: https://github.com/rust-lang/rfcs/blob/master/text/2115-argument-lifetimes.md
    /// [issue #44752]: https://github.com/rust-lang/rust/issues/44752
    pub SINGLE_USE_LIFETIMES,
    Allow,
    "detects lifetime parameters that are only used once"
}

declare_lint! {
    /// The `unused_lifetimes` lint detects lifetime parameters that are never
    /// used.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #[deny(unused_lifetimes)]
    ///
    /// pub fn foo<'a>() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Unused lifetime parameters may signal a mistake or unfinished code.
    /// Consider removing the parameter.
    pub UNUSED_LIFETIMES,
    Allow,
    "detects lifetime parameters that are never used"
}

declare_lint! {
    /// The `tyvar_behind_raw_pointer` lint detects raw pointer to an
    /// inference variable.
    ///
    /// ### Example
    ///
    /// ```rust,edition2015
    /// // edition 2015
    /// let data = std::ptr::null();
    /// let _ = &data as *const *const ();
    ///
    /// if data.is_null() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// This kind of inference was previously allowed, but with the future
    /// arrival of [arbitrary self types], this can introduce ambiguity. To
    /// resolve this, use an explicit type instead of relying on type
    /// inference.
    ///
    /// This is a [future-incompatible] lint to transition this to a hard
    /// error in the 2018 edition. See [issue #46906] for more details. This
    /// is currently a hard-error on the 2018 edition, and is "warn" by
    /// default in the 2015 edition.
    ///
    /// [arbitrary self types]: https://github.com/rust-lang/rust/issues/44874
    /// [issue #46906]: https://github.com/rust-lang/rust/issues/46906
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub TYVAR_BEHIND_RAW_POINTER,
    Warn,
    "raw pointer to an inference variable",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #46906 <https://github.com/rust-lang/rust/issues/46906>",
        edition: Some(Edition::Edition2018),
    };
}

declare_lint! {
    /// The `elided_lifetimes_in_paths` lint detects the use of hidden
    /// lifetime parameters.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(elided_lifetimes_in_paths)]
    /// struct Foo<'a> {
    ///     x: &'a u32
    /// }
    ///
    /// fn foo(x: &Foo) {
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Elided lifetime parameters can make it difficult to see at a glance
    /// that borrowing is occurring. This lint ensures that lifetime
    /// parameters are always explicitly stated, even if it is the `'_`
    /// [placeholder lifetime].
    ///
    /// This lint is "allow" by default because it has some known issues, and
    /// may require a significant transition for old code.
    ///
    /// [placeholder lifetime]: https://doc.rust-lang.org/reference/lifetime-elision.html#lifetime-elision-in-functions
    pub ELIDED_LIFETIMES_IN_PATHS,
    Allow,
    "hidden lifetime parameters in types are deprecated",
    crate_level_only
}

declare_lint! {
    /// The `bare_trait_objects` lint suggests using `dyn Trait` for trait
    /// objects.
    ///
    /// ### Example
    ///
    /// ```rust
    /// trait Trait { }
    ///
    /// fn takes_trait_object(_: Box<Trait>) {
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Without the `dyn` indicator, it can be ambiguous or confusing when
    /// reading code as to whether or not you are looking at a trait object.
    /// The `dyn` keyword makes it explicit, and adds a symmetry to contrast
    /// with [`impl Trait`].
    ///
    /// [`impl Trait`]: https://doc.rust-lang.org/book/ch10-02-traits.html#traits-as-parameters
    pub BARE_TRAIT_OBJECTS,
    Warn,
    "suggest using `dyn Trait` for trait objects"
}

declare_lint! {
    /// The `absolute_paths_not_starting_with_crate` lint detects fully
    /// qualified paths that start with a module name instead of `crate`,
    /// `self`, or an extern crate name
    ///
    /// ### Example
    ///
    /// ```rust,edition2015,compile_fail
    /// #![deny(absolute_paths_not_starting_with_crate)]
    ///
    /// mod foo {
    ///     pub fn bar() {}
    /// }
    ///
    /// fn main() {
    ///     ::foo::bar();
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Rust [editions] allow the language to evolve without breaking
    /// backwards compatibility. This lint catches code that uses absolute
    /// paths in the style of the 2015 edition. In the 2015 edition, absolute
    /// paths (those starting with `::`) refer to either the crate root or an
    /// external crate. In the 2018 edition it was changed so that they only
    /// refer to external crates. The path prefix `crate::` should be used
    /// instead to reference items from the crate root.
    ///
    /// If you switch the compiler from the 2015 to 2018 edition without
    /// updating the code, then it will fail to compile if the old style paths
    /// are used. You can manually change the paths to use the `crate::`
    /// prefix to transition to the 2018 edition.
    ///
    /// This lint solves the problem automatically. It is "allow" by default
    /// because the code is perfectly valid in the 2015 edition. The [`cargo
    /// fix`] tool with the `--edition` flag will switch this lint to "warn"
    /// and automatically apply the suggested fix from the compiler. This
    /// provides a completely automated way to update old code to the 2018
    /// edition.
    ///
    /// [editions]: https://doc.rust-lang.org/edition-guide/
    /// [`cargo fix`]: https://doc.rust-lang.org/cargo/commands/cargo-fix.html
    pub ABSOLUTE_PATHS_NOT_STARTING_WITH_CRATE,
    Allow,
    "fully qualified paths that start with a module name \
     instead of `crate`, `self`, or an extern crate name",
     @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #53130 <https://github.com/rust-lang/rust/issues/53130>",
        edition: Some(Edition::Edition2018),
     };
}

declare_lint! {
    /// The `illegal_floating_point_literal_pattern` lint detects
    /// floating-point literals used in patterns.
    ///
    /// ### Example
    ///
    /// ```rust
    /// let x = 42.0;
    ///
    /// match x {
    ///     5.0 => {}
    ///     _ => {}
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Previous versions of the compiler accepted floating-point literals in
    /// patterns, but it was later determined this was a mistake. The
    /// semantics of comparing floating-point values may not be clear in a
    /// pattern when contrasted with "structural equality". Typically you can
    /// work around this by using a [match guard], such as:
    ///
    /// ```rust
    /// # let x = 42.0;
    ///
    /// match x {
    ///     y if y == 5.0 => {}
    ///     _ => {}
    /// }
    /// ```
    ///
    /// This is a [future-incompatible] lint to transition this to a hard
    /// error in the future. See [issue #41620] for more details.
    ///
    /// [issue #41620]: https://github.com/rust-lang/rust/issues/41620
    /// [match guard]: https://doc.rust-lang.org/reference/expressions/match-expr.html#match-guards
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub ILLEGAL_FLOATING_POINT_LITERAL_PATTERN,
    Warn,
    "floating-point literals cannot be used in patterns",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #41620 <https://github.com/rust-lang/rust/issues/41620>",
        edition: None,
    };
}

declare_lint! {
    /// The `unstable_name_collisions` lint detects that you have used a name
    /// that the standard library plans to add in the future.
    ///
    /// ### Example
    ///
    /// ```rust
    /// trait MyIterator : Iterator {
    ///     // is_sorted is an unstable method that already exists on the Iterator trait
    ///     fn is_sorted(self) -> bool where Self: Sized {true}
    /// }
    ///
    /// impl<T: ?Sized> MyIterator for T where T: Iterator { }
    ///
    /// let x = vec![1, 2, 3];
    /// let _ = x.iter().is_sorted();
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// When new methods are added to traits in the standard library, they are
    /// usually added in an "unstable" form which is only available on the
    /// [nightly channel] with a [`feature` attribute]. If there is any
    /// pre-existing code which extends a trait to have a method with the same
    /// name, then the names will collide. In the future, when the method is
    /// stabilized, this will cause an error due to the ambiguity. This lint
    /// is an early-warning to let you know that there may be a collision in
    /// the future. This can be avoided by adding type annotations to
    /// disambiguate which trait method you intend to call, such as
    /// `MyIterator::is_sorted(my_iter)` or renaming or removing the method.
    ///
    /// [nightly channel]: https://doc.rust-lang.org/book/appendix-07-nightly-rust.html
    /// [`feature` attribute]: https://doc.rust-lang.org/nightly/unstable-book/
    pub UNSTABLE_NAME_COLLISIONS,
    Warn,
    "detects name collision with an existing but unstable method",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #48919 <https://github.com/rust-lang/rust/issues/48919>",
        edition: None,
        // Note: this item represents future incompatibility of all unstable functions in the
        //       standard library, and thus should never be removed or changed to an error.
    };
}

declare_lint! {
    /// The `irrefutable_let_patterns` lint detects detects [irrefutable
    /// patterns] in [if-let] and [while-let] statements.
    ///
    ///
    ///
    /// ### Example
    ///
    /// ```rust
    /// if let _ = 123 {
    ///     println!("always runs!");
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// There usually isn't a reason to have an irrefutable pattern in an
    /// if-let or while-let statement, because the pattern will always match
    /// successfully. A [`let`] or [`loop`] statement will suffice. However,
    /// when generating code with a macro, forbidding irrefutable patterns
    /// would require awkward workarounds in situations where the macro
    /// doesn't know if the pattern is refutable or not. This lint allows
    /// macros to accept this form, while alerting for a possibly incorrect
    /// use in normal code.
    ///
    /// See [RFC 2086] for more details.
    ///
    /// [irrefutable patterns]: https://doc.rust-lang.org/reference/patterns.html#refutability
    /// [if-let]: https://doc.rust-lang.org/reference/expressions/if-expr.html#if-let-expressions
    /// [while-let]: https://doc.rust-lang.org/reference/expressions/loop-expr.html#predicate-pattern-loops
    /// [`let`]: https://doc.rust-lang.org/reference/statements.html#let-statements
    /// [`loop`]: https://doc.rust-lang.org/reference/expressions/loop-expr.html#infinite-loops
    /// [RFC 2086]: https://github.com/rust-lang/rfcs/blob/master/text/2086-allow-if-let-irrefutables.md
    pub IRREFUTABLE_LET_PATTERNS,
    Warn,
    "detects irrefutable patterns in if-let and while-let statements"
}

declare_lint! {
    /// The `unused_labels` lint detects [labels] that are never used.
    ///
    /// [labels]: https://doc.rust-lang.org/reference/expressions/loop-expr.html#loop-labels
    ///
    /// ### Example
    ///
    /// ```rust,no_run
    /// 'unused_label: loop {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Unused labels may signal a mistake or unfinished code. To silence the
    /// warning for the individual label, prefix it with an underscore such as
    /// `'_my_label:`.
    pub UNUSED_LABELS,
    Warn,
    "detects labels that are never used"
}

declare_lint! {
    /// The `broken_intra_doc_links` lint detects failures in resolving
    /// intra-doc link targets. This is a `rustdoc` only lint, see the
    /// documentation in the [rustdoc book].
    ///
    /// [rustdoc book]: ../../../rustdoc/lints.html#broken_intra_doc_links
    pub BROKEN_INTRA_DOC_LINKS,
    Warn,
    "failures in resolving intra-doc link targets"
}

declare_lint! {
    /// This is a subset of `broken_intra_doc_links` that warns when linking from
    /// a public item to a private one. This is a `rustdoc` only lint, see the
    /// documentation in the [rustdoc book].
    ///
    /// [rustdoc book]: ../../../rustdoc/lints.html#private_intra_doc_links
    pub PRIVATE_INTRA_DOC_LINKS,
    Warn,
    "linking from a public item to a private one"
}

declare_lint! {
    /// The `invalid_codeblock_attributes` lint detects code block attributes
    /// in documentation examples that have potentially mis-typed values. This
    /// is a `rustdoc` only lint, see the documentation in the [rustdoc book].
    ///
    /// [rustdoc book]: ../../../rustdoc/lints.html#invalid_codeblock_attributes
    pub INVALID_CODEBLOCK_ATTRIBUTES,
    Warn,
    "codeblock attribute looks a lot like a known one"
}

declare_lint! {
    /// The `missing_crate_level_docs` lint detects if documentation is
    /// missing at the crate root. This is a `rustdoc` only lint, see the
    /// documentation in the [rustdoc book].
    ///
    /// [rustdoc book]: ../../../rustdoc/lints.html#missing_crate_level_docs
    pub MISSING_CRATE_LEVEL_DOCS,
    Allow,
    "detects crates with no crate-level documentation"
}

declare_lint! {
    /// The `missing_doc_code_examples` lint detects publicly-exported items
    /// without code samples in their documentation. This is a `rustdoc` only
    /// lint, see the documentation in the [rustdoc book].
    ///
    /// [rustdoc book]: ../../../rustdoc/lints.html#missing_doc_code_examples
    pub MISSING_DOC_CODE_EXAMPLES,
    Allow,
    "detects publicly-exported items without code samples in their documentation"
}

declare_lint! {
    /// The `private_doc_tests` lint detects code samples in docs of private
    /// items not documented by `rustdoc`. This is a `rustdoc` only lint, see
    /// the documentation in the [rustdoc book].
    ///
    /// [rustdoc book]: ../../../rustdoc/lints.html#private_doc_tests
    pub PRIVATE_DOC_TESTS,
    Allow,
    "detects code samples in docs of private items not documented by rustdoc"
}

declare_lint! {
    /// The `invalid_html_tags` lint detects invalid HTML tags. This is a
    /// `rustdoc` only lint, see the documentation in the [rustdoc book].
    ///
    /// [rustdoc book]: ../../../rustdoc/lints.html#invalid_html_tags
    pub INVALID_HTML_TAGS,
    Allow,
    "detects invalid HTML tags in doc comments"
}

declare_lint! {
    /// The `non_autolinks` lint detects when a URL could be written using
    /// only angle brackets. This is a `rustdoc` only lint, see the
    /// documentation in the [rustdoc book].
    ///
    /// [rustdoc book]: ../../../rustdoc/lints.html#non_autolinks
    pub NON_AUTOLINKS,
    Warn,
    "detects URLs that could be written using only angle brackets"
}

declare_lint! {
    /// The `where_clauses_object_safety` lint detects for [object safety] of
    /// [where clauses].
    ///
    /// [object safety]: https://doc.rust-lang.org/reference/items/traits.html#object-safety
    /// [where clauses]: https://doc.rust-lang.org/reference/items/generics.html#where-clauses
    ///
    /// ### Example
    ///
    /// ```rust,no_run
    /// trait Trait {}
    ///
    /// trait X { fn foo(&self) where Self: Trait; }
    ///
    /// impl X for () { fn foo(&self) {} }
    ///
    /// impl Trait for dyn X {}
    ///
    /// // Segfault at opt-level 0, SIGILL otherwise.
    /// pub fn main() { <dyn X as X>::foo(&()); }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The compiler previously allowed these object-unsafe bounds, which was
    /// incorrect. This is a [future-incompatible] lint to transition this to
    /// a hard error in the future. See [issue #51443] for more details.
    ///
    /// [issue #51443]: https://github.com/rust-lang/rust/issues/51443
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub WHERE_CLAUSES_OBJECT_SAFETY,
    Warn,
    "checks the object safety of where clauses",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #51443 <https://github.com/rust-lang/rust/issues/51443>",
        edition: None,
    };
}

declare_lint! {
    /// The `proc_macro_derive_resolution_fallback` lint detects proc macro
    /// derives using inaccessible names from parent modules.
    ///
    /// ### Example
    ///
    /// ```rust,ignore (proc-macro)
    /// // foo.rs
    /// #![crate_type = "proc-macro"]
    ///
    /// extern crate proc_macro;
    ///
    /// use proc_macro::*;
    ///
    /// #[proc_macro_derive(Foo)]
    /// pub fn foo1(a: TokenStream) -> TokenStream {
    ///     drop(a);
    ///     "mod __bar { static mut BAR: Option<Something> = None; }".parse().unwrap()
    /// }
    /// ```
    ///
    /// ```rust,ignore (needs-dependency)
    /// // bar.rs
    /// #[macro_use]
    /// extern crate foo;
    ///
    /// struct Something;
    ///
    /// #[derive(Foo)]
    /// struct Another;
    ///
    /// fn main() {}
    /// ```
    ///
    /// This will produce:
    ///
    /// ```text
    /// warning: cannot find type `Something` in this scope
    ///  --> src/main.rs:8:10
    ///   |
    /// 8 | #[derive(Foo)]
    ///   |          ^^^ names from parent modules are not accessible without an explicit import
    ///   |
    ///   = note: `#[warn(proc_macro_derive_resolution_fallback)]` on by default
    ///   = warning: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    ///   = note: for more information, see issue #50504 <https://github.com/rust-lang/rust/issues/50504>
    /// ```
    ///
    /// ### Explanation
    ///
    /// If a proc-macro generates a module, the compiler unintentionally
    /// allowed items in that module to refer to items in the crate root
    /// without importing them. This is a [future-incompatible] lint to
    /// transition this to a hard error in the future. See [issue #50504] for
    /// more details.
    ///
    /// [issue #50504]: https://github.com/rust-lang/rust/issues/50504
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub PROC_MACRO_DERIVE_RESOLUTION_FALLBACK,
    Warn,
    "detects proc macro derives using inaccessible names from parent modules",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #50504 <https://github.com/rust-lang/rust/issues/50504>",
        edition: None,
    };
}

declare_lint! {
    /// The `macro_use_extern_crate` lint detects the use of the
    /// [`macro_use` attribute].
    ///
    /// ### Example
    ///
    /// ```rust,ignore (needs extern crate)
    /// #![deny(macro_use_extern_crate)]
    ///
    /// #[macro_use]
    /// extern crate serde_json;
    ///
    /// fn main() {
    ///     let _ = json!{{}};
    /// }
    /// ```
    ///
    /// This will produce:
    ///
    /// ```text
    /// error: deprecated `#[macro_use]` attribute used to import macros should be replaced at use sites with a `use` item to import the macro instead
    ///  --> src/main.rs:3:1
    ///   |
    /// 3 | #[macro_use]
    ///   | ^^^^^^^^^^^^
    ///   |
    /// note: the lint level is defined here
    ///  --> src/main.rs:1:9
    ///   |
    /// 1 | #![deny(macro_use_extern_crate)]
    ///   |         ^^^^^^^^^^^^^^^^^^^^^^
    /// ```
    ///
    /// ### Explanation
    ///
    /// The [`macro_use` attribute] on an [`extern crate`] item causes
    /// macros in that external crate to be brought into the prelude of the
    /// crate, making the macros in scope everywhere. As part of the efforts
    /// to simplify handling of dependencies in the [2018 edition], the use of
    /// `extern crate` is being phased out. To bring macros from extern crates
    /// into scope, it is recommended to use a [`use` import].
    ///
    /// This lint is "allow" by default because this is a stylistic choice
    /// that has not been settled, see [issue #52043] for more information.
    ///
    /// [`macro_use` attribute]: https://doc.rust-lang.org/reference/macros-by-example.html#the-macro_use-attribute
    /// [`use` import]: https://doc.rust-lang.org/reference/items/use-declarations.html
    /// [issue #52043]: https://github.com/rust-lang/rust/issues/52043
    pub MACRO_USE_EXTERN_CRATE,
    Allow,
    "the `#[macro_use]` attribute is now deprecated in favor of using macros \
     via the module system"
}

declare_lint! {
    /// The `macro_expanded_macro_exports_accessed_by_absolute_paths` lint
    /// detects macro-expanded [`macro_export`] macros from the current crate
    /// that cannot be referred to by absolute paths.
    ///
    /// [`macro_export`]: https://doc.rust-lang.org/reference/macros-by-example.html#path-based-scope
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// macro_rules! define_exported {
    ///     () => {
    ///         #[macro_export]
    ///         macro_rules! exported {
    ///             () => {};
    ///         }
    ///     };
    /// }
    ///
    /// define_exported!();
    ///
    /// fn main() {
    ///     crate::exported!();
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The intent is that all macros marked with the `#[macro_export]`
    /// attribute are made available in the root of the crate. However, when a
    /// `macro_rules!` definition is generated by another macro, the macro
    /// expansion is unable to uphold this rule. This is a
    /// [future-incompatible] lint to transition this to a hard error in the
    /// future. See [issue #53495] for more details.
    ///
    /// [issue #53495]: https://github.com/rust-lang/rust/issues/53495
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub MACRO_EXPANDED_MACRO_EXPORTS_ACCESSED_BY_ABSOLUTE_PATHS,
    Deny,
    "macro-expanded `macro_export` macros from the current crate \
     cannot be referred to by absolute paths",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #52234 <https://github.com/rust-lang/rust/issues/52234>",
        edition: None,
    };
    crate_level_only
}

declare_lint! {
    /// The `explicit_outlives_requirements` lint detects unnecessary
    /// lifetime bounds that can be inferred.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// # #![allow(unused)]
    /// #![deny(explicit_outlives_requirements)]
    ///
    /// struct SharedRef<'a, T>
    /// where
    ///     T: 'a,
    /// {
    ///     data: &'a T,
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// If a `struct` contains a reference, such as `&'a T`, the compiler
    /// requires that `T` outlives the lifetime `'a`. This historically
    /// required writing an explicit lifetime bound to indicate this
    /// requirement. However, this can be overly explicit, causing clutter and
    /// unnecessary complexity. The language was changed to automatically
    /// infer the bound if it is not specified. Specifically, if the struct
    /// contains a reference, directly or indirectly, to `T` with lifetime
    /// `'x`, then it will infer that `T: 'x` is a requirement.
    ///
    /// This lint is "allow" by default because it can be noisy for existing
    /// code that already had these requirements. This is a stylistic choice,
    /// as it is still valid to explicitly state the bound. It also has some
    /// false positives that can cause confusion.
    ///
    /// See [RFC 2093] for more details.
    ///
    /// [RFC 2093]: https://github.com/rust-lang/rfcs/blob/master/text/2093-infer-outlives.md
    pub EXPLICIT_OUTLIVES_REQUIREMENTS,
    Allow,
    "outlives requirements can be inferred"
}

declare_lint! {
    /// The `indirect_structural_match` lint detects a `const` in a pattern
    /// that manually implements [`PartialEq`] and [`Eq`].
    ///
    /// [`PartialEq`]: https://doc.rust-lang.org/std/cmp/trait.PartialEq.html
    /// [`Eq`]: https://doc.rust-lang.org/std/cmp/trait.Eq.html
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(indirect_structural_match)]
    ///
    /// struct NoDerive(i32);
    /// impl PartialEq for NoDerive { fn eq(&self, _: &Self) -> bool { false } }
    /// impl Eq for NoDerive { }
    /// #[derive(PartialEq, Eq)]
    /// struct WrapParam<T>(T);
    /// const WRAP_INDIRECT_PARAM: & &WrapParam<NoDerive> = & &WrapParam(NoDerive(0));
    /// fn main() {
    ///     match WRAP_INDIRECT_PARAM {
    ///         WRAP_INDIRECT_PARAM => { }
    ///         _ => { }
    ///     }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The compiler unintentionally accepted this form in the past. This is a
    /// [future-incompatible] lint to transition this to a hard error in the
    /// future. See [issue #62411] for a complete description of the problem,
    /// and some possible solutions.
    ///
    /// [issue #62411]: https://github.com/rust-lang/rust/issues/62411
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub INDIRECT_STRUCTURAL_MATCH,
    Warn,
    "constant used in pattern contains value of non-structural-match type in a field or a variant",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #62411 <https://github.com/rust-lang/rust/issues/62411>",
        edition: None,
    };
}

declare_lint! {
    /// The `deprecated_in_future` lint is internal to rustc and should not be
    /// used by user code.
    ///
    /// This lint is only enabled in the standard library. It works with the
    /// use of `#[rustc_deprecated]` with a `since` field of a version in the
    /// future. This allows something to be marked as deprecated in a future
    /// version, and then this lint will ensure that the item is no longer
    /// used in the standard library. See the [stability documentation] for
    /// more details.
    ///
    /// [stability documentation]: https://rustc-dev-guide.rust-lang.org/stability.html#rustc_deprecated
    pub DEPRECATED_IN_FUTURE,
    Allow,
    "detects use of items that will be deprecated in a future version",
    report_in_external_macro
}

declare_lint! {
    /// The `pointer_structural_match` lint detects pointers used in patterns whose behaviour
    /// cannot be relied upon across compiler versions and optimization levels.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(pointer_structural_match)]
    /// fn foo(a: usize, b: usize) -> usize { a + b }
    /// const FOO: fn(usize, usize) -> usize = foo;
    /// fn main() {
    ///     match FOO {
    ///         FOO => {},
    ///         _ => {},
    ///     }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Previous versions of Rust allowed function pointers and wide raw pointers in patterns.
    /// While these work in many cases as expected by users, it is possible that due to
    /// optimizations pointers are "not equal to themselves" or pointers to different functions
    /// compare as equal during runtime. This is because LLVM optimizations can deduplicate
    /// functions if their bodies are the same, thus also making pointers to these functions point
    /// to the same location. Additionally functions may get duplicated if they are instantiated
    /// in different crates and not deduplicated again via LTO.
    pub POINTER_STRUCTURAL_MATCH,
    Allow,
    "pointers are not structural-match",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #62411 <https://github.com/rust-lang/rust/issues/70861>",
        edition: None,
    };
}

declare_lint! {
    /// The `nontrivial_structural_match` lint detects constants that are used in patterns,
    /// whose type is not structural-match and whose initializer body actually uses values
    /// that are not structural-match. So `Option<NotStruturalMatch>` is ok if the constant
    /// is just `None`.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(nontrivial_structural_match)]
    ///
    /// #[derive(Copy, Clone, Debug)]
    /// struct NoDerive(u32);
    /// impl PartialEq for NoDerive { fn eq(&self, _: &Self) -> bool { false } }
    /// impl Eq for NoDerive { }
    /// fn main() {
    ///     const INDEX: Option<NoDerive> = [None, Some(NoDerive(10))][0];
    ///     match None { Some(_) => panic!("whoops"), INDEX => dbg!(INDEX), };
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Previous versions of Rust accepted constants in patterns, even if those constants's types
    /// did not have `PartialEq` derived. Thus the compiler falls back to runtime execution of
    /// `PartialEq`, which can report that two constants are not equal even if they are
    /// bit-equivalent.
    pub NONTRIVIAL_STRUCTURAL_MATCH,
    Warn,
    "constant used in pattern of non-structural-match type and the constant's initializer \
    expression contains values of non-structural-match types",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #73448 <https://github.com/rust-lang/rust/issues/73448>",
        edition: None,
    };
}

declare_lint! {
    /// The `ambiguous_associated_items` lint detects ambiguity between
    /// [associated items] and [enum variants].
    ///
    /// [associated items]: https://doc.rust-lang.org/reference/items/associated-items.html
    /// [enum variants]: https://doc.rust-lang.org/reference/items/enumerations.html
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// enum E {
    ///     V
    /// }
    ///
    /// trait Tr {
    ///     type V;
    ///     fn foo() -> Self::V;
    /// }
    ///
    /// impl Tr for E {
    ///     type V = u8;
    ///     // `Self::V` is ambiguous because it may refer to the associated type or
    ///     // the enum variant.
    ///     fn foo() -> Self::V { 0 }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Previous versions of Rust did not allow accessing enum variants
    /// through [type aliases]. When this ability was added (see [RFC 2338]), this
    /// introduced some situations where it can be ambiguous what a type
    /// was referring to.
    ///
    /// To fix this ambiguity, you should use a [qualified path] to explicitly
    /// state which type to use. For example, in the above example the
    /// function can be written as `fn f() -> <Self as Tr>::V { 0 }` to
    /// specifically refer to the associated type.
    ///
    /// This is a [future-incompatible] lint to transition this to a hard
    /// error in the future. See [issue #57644] for more details.
    ///
    /// [issue #57644]: https://github.com/rust-lang/rust/issues/57644
    /// [type aliases]: https://doc.rust-lang.org/reference/items/type-aliases.html#type-aliases
    /// [RFC 2338]: https://github.com/rust-lang/rfcs/blob/master/text/2338-type-alias-enum-variants.md
    /// [qualified path]: https://doc.rust-lang.org/reference/paths.html#qualified-paths
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub AMBIGUOUS_ASSOCIATED_ITEMS,
    Deny,
    "ambiguous associated items",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #57644 <https://github.com/rust-lang/rust/issues/57644>",
        edition: None,
    };
}

declare_lint! {
    /// The `mutable_borrow_reservation_conflict` lint detects the reservation
    /// of a two-phased borrow that conflicts with other shared borrows.
    ///
    /// ### Example
    ///
    /// ```rust
    /// let mut v = vec![0, 1, 2];
    /// let shared = &v;
    /// v.push(shared.len());
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// This is a [future-incompatible] lint to transition this to a hard error
    /// in the future. See [issue #59159] for a complete description of the
    /// problem, and some possible solutions.
    ///
    /// [issue #59159]: https://github.com/rust-lang/rust/issues/59159
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub MUTABLE_BORROW_RESERVATION_CONFLICT,
    Warn,
    "reservation of a two-phased borrow conflicts with other shared borrows",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #59159 <https://github.com/rust-lang/rust/issues/59159>",
        edition: None,
    };
}

declare_lint! {
    /// The `soft_unstable` lint detects unstable features that were
    /// unintentionally allowed on stable.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #[cfg(test)]
    /// extern crate test;
    ///
    /// #[bench]
    /// fn name(b: &mut test::Bencher) {
    ///     b.iter(|| 123)
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The [`bench` attribute] was accidentally allowed to be specified on
    /// the [stable release channel]. Turning this to a hard error would have
    /// broken some projects. This lint allows those projects to continue to
    /// build correctly when [`--cap-lints`] is used, but otherwise signal an
    /// error that `#[bench]` should not be used on the stable channel. This
    /// is a [future-incompatible] lint to transition this to a hard error in
    /// the future. See [issue #64266] for more details.
    ///
    /// [issue #64266]: https://github.com/rust-lang/rust/issues/64266
    /// [`bench` attribute]: https://doc.rust-lang.org/nightly/unstable-book/library-features/test.html
    /// [stable release channel]: https://doc.rust-lang.org/book/appendix-07-nightly-rust.html
    /// [`--cap-lints`]: https://doc.rust-lang.org/rustc/lints/levels.html#capping-lints
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub SOFT_UNSTABLE,
    Deny,
    "a feature gate that doesn't break dependent crates",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #64266 <https://github.com/rust-lang/rust/issues/64266>",
        edition: None,
    };
}

declare_lint! {
    /// The `inline_no_sanitize` lint detects incompatible use of
    /// [`#[inline(always)]`][inline] and [`#[no_sanitize(...)]`][no_sanitize].
    ///
    /// [inline]: https://doc.rust-lang.org/reference/attributes/codegen.html#the-inline-attribute
    /// [no_sanitize]: https://doc.rust-lang.org/nightly/unstable-book/language-features/no-sanitize.html
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![feature(no_sanitize)]
    ///
    /// #[inline(always)]
    /// #[no_sanitize(address)]
    /// fn x() {}
    ///
    /// fn main() {
    ///     x()
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The use of the [`#[inline(always)]`][inline] attribute prevents the
    /// the [`#[no_sanitize(...)]`][no_sanitize] attribute from working.
    /// Consider temporarily removing `inline` attribute.
    pub INLINE_NO_SANITIZE,
    Warn,
    "detects incompatible use of `#[inline(always)]` and `#[no_sanitize(...)]`",
}

declare_lint! {
    /// The `asm_sub_register` lint detects using only a subset of a register
    /// for inline asm inputs.
    ///
    /// ### Example
    ///
    /// ```rust,ignore (fails on system llvm)
    /// #![feature(asm)]
    ///
    /// fn main() {
    ///     #[cfg(target_arch="x86_64")]
    ///     unsafe {
    ///         asm!("mov {0}, {0}", in(reg) 0i16);
    ///     }
    /// }
    /// ```
    ///
    /// This will produce:
    ///
    /// ```text
    /// warning: formatting may not be suitable for sub-register argument
    ///  --> src/main.rs:6:19
    ///   |
    /// 6 |         asm!("mov {0}, {0}", in(reg) 0i16);
    ///   |                   ^^^  ^^^           ---- for this argument
    ///   |
    ///   = note: `#[warn(asm_sub_register)]` on by default
    ///   = help: use the `x` modifier to have the register formatted as `ax`
    ///   = help: or use the `r` modifier to keep the default formatting of `rax`
    /// ```
    ///
    /// ### Explanation
    ///
    /// Registers on some architectures can use different names to refer to a
    /// subset of the register. By default, the compiler will use the name for
    /// the full register size. To explicitly use a subset of the register,
    /// you can override the default by using a modifier on the template
    /// string operand to specify when subregister to use. This lint is issued
    /// if you pass in a value with a smaller data type than the default
    /// register size, to alert you of possibly using the incorrect width. To
    /// fix this, add the suggested modifier to the template, or cast the
    /// value to the correct size.
    ///
    /// See [register template modifiers] for more details.
    ///
    /// [register template modifiers]: https://doc.rust-lang.org/nightly/unstable-book/library-features/asm.html#register-template-modifiers
    pub ASM_SUB_REGISTER,
    Warn,
    "using only a subset of a register for inline asm inputs",
}

declare_lint! {
    /// The `unsafe_op_in_unsafe_fn` lint detects unsafe operations in unsafe
    /// functions without an explicit unsafe block. This lint only works on
    /// the [**nightly channel**] with the
    /// `#![feature(unsafe_block_in_unsafe_fn)]` feature.
    ///
    /// [**nightly channel**]: https://doc.rust-lang.org/book/appendix-07-nightly-rust.html
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![feature(unsafe_block_in_unsafe_fn)]
    /// #![deny(unsafe_op_in_unsafe_fn)]
    ///
    /// unsafe fn foo() {}
    ///
    /// unsafe fn bar() {
    ///     foo();
    /// }
    ///
    /// fn main() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Currently, an [`unsafe fn`] allows any [unsafe] operation within its
    /// body. However, this can increase the surface area of code that needs
    /// to be scrutinized for proper behavior. The [`unsafe` block] provides a
    /// convenient way to make it clear exactly which parts of the code are
    /// performing unsafe operations. In the future, it is desired to change
    /// it so that unsafe operations cannot be performed in an `unsafe fn`
    /// without an `unsafe` block.
    ///
    /// The fix to this is to wrap the unsafe code in an `unsafe` block.
    ///
    /// This lint is "allow" by default because it has not yet been
    /// stabilized, and is not yet complete. See [RFC #2585] and [issue
    /// #71668] for more details
    ///
    /// [`unsafe fn`]: https://doc.rust-lang.org/reference/unsafe-functions.html
    /// [`unsafe` block]: https://doc.rust-lang.org/reference/expressions/block-expr.html#unsafe-blocks
    /// [unsafe]: https://doc.rust-lang.org/reference/unsafety.html
    /// [RFC #2585]: https://github.com/rust-lang/rfcs/blob/master/text/2585-unsafe-block-in-unsafe-fn.md
    /// [issue #71668]: https://github.com/rust-lang/rust/issues/71668
    pub UNSAFE_OP_IN_UNSAFE_FN,
    Allow,
    "unsafe operations in unsafe functions without an explicit unsafe block are deprecated",
    @feature_gate = sym::unsafe_block_in_unsafe_fn;
}

declare_lint! {
    /// The `cenum_impl_drop_cast` lint detects an `as` cast of a field-less
    /// `enum` that implements [`Drop`].
    ///
    /// [`Drop`]: https://doc.rust-lang.org/std/ops/trait.Drop.html
    ///
    /// ### Example
    ///
    /// ```rust
    /// # #![allow(unused)]
    /// enum E {
    ///     A,
    /// }
    ///
    /// impl Drop for E {
    ///     fn drop(&mut self) {
    ///         println!("Drop");
    ///     }
    /// }
    ///
    /// fn main() {
    ///     let e = E::A;
    ///     let i = e as u32;
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Casting a field-less `enum` that does not implement [`Copy`] to an
    /// integer moves the value without calling `drop`. This can result in
    /// surprising behavior if it was expected that `drop` should be called.
    /// Calling `drop` automatically would be inconsistent with other move
    /// operations. Since neither behavior is clear or consistent, it was
    /// decided that a cast of this nature will no longer be allowed.
    ///
    /// This is a [future-incompatible] lint to transition this to a hard error
    /// in the future. See [issue #73333] for more details.
    ///
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    /// [issue #73333]: https://github.com/rust-lang/rust/issues/73333
    /// [`Copy`]: https://doc.rust-lang.org/std/marker/trait.Copy.html
    pub CENUM_IMPL_DROP_CAST,
    Warn,
    "a C-like enum implementing Drop is cast",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #73333 <https://github.com/rust-lang/rust/issues/73333>",
        edition: None,
    };
}

declare_lint! {
    /// The `const_evaluatable_unchecked` lint detects a generic constant used
    /// in a type.
    ///
    /// ### Example
    ///
    /// ```rust
    /// const fn foo<T>() -> usize {
    ///     if std::mem::size_of::<*mut T>() < 8 { // size of *mut T does not depend on T
    ///         4
    ///     } else {
    ///         8
    ///     }
    /// }
    ///
    /// fn test<T>() {
    ///     let _ = [0; foo::<T>()];
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// In the 1.43 release, some uses of generic parameters in array repeat
    /// expressions were accidentally allowed. This is a [future-incompatible]
    /// lint to transition this to a hard error in the future. See [issue
    /// #76200] for a more detailed description and possible fixes.
    ///
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    /// [issue #76200]: https://github.com/rust-lang/rust/issues/76200
    pub CONST_EVALUATABLE_UNCHECKED,
    Warn,
    "detects a generic constant is used in a type without a emitting a warning",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #76200 <https://github.com/rust-lang/rust/issues/76200>",
        edition: None,
    };
}

declare_lint! {
    /// The `function_item_references` lint detects function references that are
    /// formatted with [`fmt::Pointer`] or transmuted.
    ///
    /// [`fmt::Pointer`]: https://doc.rust-lang.org/std/fmt/trait.Pointer.html
    ///
    /// ### Example
    ///
    /// ```rust
    /// fn foo() { }
    ///
    /// fn main() {
    ///     println!("{:p}", &foo);
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Taking a reference to a function may be mistaken as a way to obtain a
    /// pointer to that function. This can give unexpected results when
    /// formatting the reference as a pointer or transmuting it. This lint is
    /// issued when function references are formatted as pointers, passed as
    /// arguments bound by [`fmt::Pointer`] or transmuted.
    pub FUNCTION_ITEM_REFERENCES,
    Warn,
    "suggest casting to a function pointer when attempting to take references to function items",
}

declare_lint! {
    /// The `uninhabited_static` lint detects uninhabited statics.
    ///
    /// ### Example
    ///
    /// ```rust
    /// enum Void {}
    /// extern {
    ///     static EXTERN: Void;
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Statics with an uninhabited type can never be initialized, so they are impossible to define.
    /// However, this can be side-stepped with an `extern static`, leading to problems later in the
    /// compiler which assumes that there are no initialized uninhabited places (such as locals or
    /// statics). This was accientally allowed, but is being phased out.
    pub UNINHABITED_STATIC,
    Warn,
    "uninhabited static",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #74840 <https://github.com/rust-lang/rust/issues/74840>",
        edition: None,
    };
}

declare_lint! {
    /// The `useless_deprecated` lint detects deprecation attributes with no effect.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// struct X;
    ///
    /// #[deprecated = "message"]
    /// impl Default for X {
    ///     fn default() -> Self {
    ///         X
    ///     }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Deprecation attributes have no effect on trait implementations.
    pub USELESS_DEPRECATED,
    Deny,
    "detects deprecation attributes with no effect",
}

declare_lint! {
    /// The `unsupported_naked_functions` lint detects naked function
    /// definitions that are unsupported but were previously accepted.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![feature(naked_functions)]
    ///
    /// #[naked]
    /// pub fn f() -> u32 {
    ///     42
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The naked functions must be defined using a single inline assembly
    /// block.
    ///
    /// The execution must never fall through past the end of the assembly
    /// code so the block must use `noreturn` option. The asm block can also
    /// use `att_syntax` option, but other options are not allowed.
    ///
    /// The asm block must not contain any operands other than `const` and
    /// `sym`. Additionally, naked function should specify a non-Rust ABI.
    ///
    /// While other definitions of naked functions were previously accepted,
    /// they are unsupported and might not work reliably. This is a
    /// [future-incompatible] lint that will transition into hard error in
    /// the future.
    ///
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub UNSUPPORTED_NAKED_FUNCTIONS,
    Warn,
    "unsupported naked function definitions",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #32408 <https://github.com/rust-lang/rust/issues/32408>",
        edition: None,
    };
}

declare_tool_lint! {
    pub rustc::INEFFECTIVE_UNSTABLE_TRAIT_IMPL,
    Deny,
    "detects `#[unstable]` on stable trait implementations for stable types"
}

declare_lint_pass! {
    /// Does nothing as a lint pass, but registers some `Lint`s
    /// that are used by other parts of the compiler.
    HardwiredLints => [
        ILLEGAL_FLOATING_POINT_LITERAL_PATTERN,
        ARITHMETIC_OVERFLOW,
        UNCONDITIONAL_PANIC,
        UNUSED_IMPORTS,
        UNUSED_EXTERN_CRATES,
        UNUSED_CRATE_DEPENDENCIES,
        UNUSED_QUALIFICATIONS,
        UNKNOWN_LINTS,
        UNUSED_VARIABLES,
        UNUSED_ASSIGNMENTS,
        DEAD_CODE,
        UNREACHABLE_CODE,
        UNREACHABLE_PATTERNS,
        OVERLAPPING_RANGE_ENDPOINTS,
        BINDINGS_WITH_VARIANT_NAME,
        UNUSED_MACROS,
        WARNINGS,
        UNUSED_FEATURES,
        STABLE_FEATURES,
        UNKNOWN_CRATE_TYPES,
        TRIVIAL_CASTS,
        TRIVIAL_NUMERIC_CASTS,
        PRIVATE_IN_PUBLIC,
        EXPORTED_PRIVATE_DEPENDENCIES,
        PUB_USE_OF_PRIVATE_EXTERN_CRATE,
        INVALID_TYPE_PARAM_DEFAULT,
        CONST_ERR,
        RENAMED_AND_REMOVED_LINTS,
        UNALIGNED_REFERENCES,
        CONST_ITEM_MUTATION,
        SAFE_PACKED_BORROWS,
        PATTERNS_IN_FNS_WITHOUT_BODY,
        MISSING_FRAGMENT_SPECIFIER,
        LATE_BOUND_LIFETIME_ARGUMENTS,
        ORDER_DEPENDENT_TRAIT_OBJECTS,
        COHERENCE_LEAK_CHECK,
        DEPRECATED,
        UNUSED_UNSAFE,
        UNUSED_MUT,
        UNCONDITIONAL_RECURSION,
        SINGLE_USE_LIFETIMES,
        UNUSED_LIFETIMES,
        UNUSED_LABELS,
        TYVAR_BEHIND_RAW_POINTER,
        ELIDED_LIFETIMES_IN_PATHS,
        BARE_TRAIT_OBJECTS,
        ABSOLUTE_PATHS_NOT_STARTING_WITH_CRATE,
        UNSTABLE_NAME_COLLISIONS,
        IRREFUTABLE_LET_PATTERNS,
        BROKEN_INTRA_DOC_LINKS,
        PRIVATE_INTRA_DOC_LINKS,
        INVALID_CODEBLOCK_ATTRIBUTES,
        MISSING_CRATE_LEVEL_DOCS,
        MISSING_DOC_CODE_EXAMPLES,
        INVALID_HTML_TAGS,
        PRIVATE_DOC_TESTS,
        NON_AUTOLINKS,
        WHERE_CLAUSES_OBJECT_SAFETY,
        PROC_MACRO_DERIVE_RESOLUTION_FALLBACK,
        MACRO_USE_EXTERN_CRATE,
        MACRO_EXPANDED_MACRO_EXPORTS_ACCESSED_BY_ABSOLUTE_PATHS,
        ILL_FORMED_ATTRIBUTE_INPUT,
        CONFLICTING_REPR_HINTS,
        META_VARIABLE_MISUSE,
        DEPRECATED_IN_FUTURE,
        AMBIGUOUS_ASSOCIATED_ITEMS,
        MUTABLE_BORROW_RESERVATION_CONFLICT,
        INDIRECT_STRUCTURAL_MATCH,
        POINTER_STRUCTURAL_MATCH,
        NONTRIVIAL_STRUCTURAL_MATCH,
        SOFT_UNSTABLE,
        INLINE_NO_SANITIZE,
        ASM_SUB_REGISTER,
        UNSAFE_OP_IN_UNSAFE_FN,
        INCOMPLETE_INCLUDE,
        CENUM_IMPL_DROP_CAST,
        CONST_EVALUATABLE_UNCHECKED,
        INEFFECTIVE_UNSTABLE_TRAIT_IMPL,
        UNINHABITED_STATIC,
        FUNCTION_ITEM_REFERENCES,
        USELESS_DEPRECATED,
        UNSUPPORTED_NAKED_FUNCTIONS,
    ]
}

declare_lint! {
    /// The `unused_doc_comments` lint detects doc comments that aren't used
    /// by `rustdoc`.
    ///
    /// ### Example
    ///
    /// ```rust
    /// /// docs for x
    /// let x = 12;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// `rustdoc` does not use doc comments in all positions, and so the doc
    /// comment will be ignored. Try changing it to a normal comment with `//`
    /// to avoid the warning.
    pub UNUSED_DOC_COMMENTS,
    Warn,
    "detects doc comments that aren't used by rustdoc"
}

declare_lint_pass!(UnusedDocComment => [UNUSED_DOC_COMMENTS]);
