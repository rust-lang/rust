//! Some lints that are built in to the compiler.
//!
//! These are the built-in lints that are emitted direct in the main
//! compiler code, rather than using their own custom pass. Those
//! lints are all available in `rustc_lint::builtin`.
//!
//! When removing a lint, make sure to also add a call to `register_removed` in
//! compiler/rustc_lint/src/lib.rs.

use rustc_span::edition::Edition;

use crate::{FutureIncompatibilityReason, declare_lint, declare_lint_pass};

declare_lint_pass! {
    /// Does nothing as a lint pass, but registers some `Lint`s
    /// that are used by other parts of the compiler.
    HardwiredLints => [
        // tidy-alphabetical-start
        AARCH64_SOFTFLOAT_NEON,
        ABSOLUTE_PATHS_NOT_STARTING_WITH_CRATE,
        AMBIGUOUS_ASSOCIATED_ITEMS,
        AMBIGUOUS_GLOB_IMPORTS,
        AMBIGUOUS_GLOB_REEXPORTS,
        ARITHMETIC_OVERFLOW,
        ASM_SUB_REGISTER,
        BAD_ASM_STYLE,
        BARE_TRAIT_OBJECTS,
        BINDINGS_WITH_VARIANT_NAME,
        BREAK_WITH_LABEL_AND_LOOP,
        COHERENCE_LEAK_CHECK,
        CONFLICTING_REPR_HINTS,
        CONST_EVALUATABLE_UNCHECKED,
        CONST_ITEM_MUTATION,
        DEAD_CODE,
        DEPENDENCY_ON_UNIT_NEVER_TYPE_FALLBACK,
        DEPRECATED,
        DEPRECATED_IN_FUTURE,
        DEPRECATED_SAFE_2024,
        DEPRECATED_WHERE_CLAUSE_LOCATION,
        DUPLICATE_MACRO_ATTRIBUTES,
        ELIDED_LIFETIMES_IN_ASSOCIATED_CONSTANT,
        ELIDED_LIFETIMES_IN_PATHS,
        ELIDED_NAMED_LIFETIMES,
        EXPLICIT_BUILTIN_CFGS_IN_FLAGS,
        EXPORTED_PRIVATE_DEPENDENCIES,
        FFI_UNWIND_CALLS,
        FORBIDDEN_LINT_GROUPS,
        FUNCTION_ITEM_REFERENCES,
        FUZZY_PROVENANCE_CASTS,
        HIDDEN_GLOB_REEXPORTS,
        ILL_FORMED_ATTRIBUTE_INPUT,
        INCOMPLETE_INCLUDE,
        INEFFECTIVE_UNSTABLE_TRAIT_IMPL,
        INLINE_NO_SANITIZE,
        INVALID_DOC_ATTRIBUTES,
        INVALID_MACRO_EXPORT_ARGUMENTS,
        INVALID_TYPE_PARAM_DEFAULT,
        IRREFUTABLE_LET_PATTERNS,
        LARGE_ASSIGNMENTS,
        LATE_BOUND_LIFETIME_ARGUMENTS,
        LEGACY_DERIVE_HELPERS,
        LINKER_MESSAGES,
        LONG_RUNNING_CONST_EVAL,
        LOSSY_PROVENANCE_CASTS,
        MACRO_EXPANDED_MACRO_EXPORTS_ACCESSED_BY_ABSOLUTE_PATHS,
        MACRO_USE_EXTERN_CRATE,
        META_VARIABLE_MISUSE,
        MISSING_ABI,
        MISSING_FRAGMENT_SPECIFIER,
        MISSING_UNSAFE_ON_EXTERN,
        MUST_NOT_SUSPEND,
        NAMED_ARGUMENTS_USED_POSITIONALLY,
        NEVER_TYPE_FALLBACK_FLOWING_INTO_UNSAFE,
        NON_CONTIGUOUS_RANGE_ENDPOINTS,
        NON_EXHAUSTIVE_OMITTED_PATTERNS,
        OUT_OF_SCOPE_MACRO_CALLS,
        OVERLAPPING_RANGE_ENDPOINTS,
        PATTERNS_IN_FNS_WITHOUT_BODY,
        PRIVATE_BOUNDS,
        PRIVATE_INTERFACES,
        PROC_MACRO_DERIVE_RESOLUTION_FALLBACK,
        PUB_USE_OF_PRIVATE_EXTERN_CRATE,
        REDUNDANT_IMPORTS,
        REDUNDANT_LIFETIMES,
        REFINING_IMPL_TRAIT_INTERNAL,
        REFINING_IMPL_TRAIT_REACHABLE,
        RENAMED_AND_REMOVED_LINTS,
        REPR_TRANSPARENT_EXTERNAL_PRIVATE_FIELDS,
        RUST_2021_INCOMPATIBLE_CLOSURE_CAPTURES,
        RUST_2021_INCOMPATIBLE_OR_PATTERNS,
        RUST_2021_PREFIXES_INCOMPATIBLE_SYNTAX,
        RUST_2021_PRELUDE_COLLISIONS,
        RUST_2024_GUARDED_STRING_INCOMPATIBLE_SYNTAX,
        RUST_2024_INCOMPATIBLE_PAT,
        RUST_2024_PRELUDE_COLLISIONS,
        SELF_CONSTRUCTOR_FROM_OUTER_ITEM,
        SEMICOLON_IN_EXPRESSIONS_FROM_MACROS,
        SINGLE_USE_LIFETIMES,
        SOFT_UNSTABLE,
        STABLE_FEATURES,
        SUPERTRAIT_ITEM_SHADOWING_DEFINITION,
        SUPERTRAIT_ITEM_SHADOWING_USAGE,
        TAIL_EXPR_DROP_ORDER,
        TEST_UNSTABLE_LINT,
        TEXT_DIRECTION_CODEPOINT_IN_COMMENT,
        TRIVIAL_CASTS,
        TRIVIAL_NUMERIC_CASTS,
        TYVAR_BEHIND_RAW_POINTER,
        UNCONDITIONAL_PANIC,
        UNCONDITIONAL_RECURSION,
        UNCOVERED_PARAM_IN_PROJECTION,
        UNEXPECTED_CFGS,
        UNFULFILLED_LINT_EXPECTATIONS,
        UNINHABITED_STATIC,
        UNKNOWN_CRATE_TYPES,
        UNKNOWN_LINTS,
        UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES,
        UNNAMEABLE_TEST_ITEMS,
        UNNAMEABLE_TYPES,
        UNREACHABLE_CODE,
        UNREACHABLE_PATTERNS,
        UNSAFE_ATTR_OUTSIDE_UNSAFE,
        UNSAFE_OP_IN_UNSAFE_FN,
        UNSTABLE_NAME_COLLISIONS,
        UNSTABLE_SYNTAX_PRE_EXPANSION,
        UNSUPPORTED_FN_PTR_CALLING_CONVENTIONS,
        UNUSED_ASSIGNMENTS,
        UNUSED_ASSOCIATED_TYPE_BOUNDS,
        UNUSED_ATTRIBUTES,
        UNUSED_CRATE_DEPENDENCIES,
        UNUSED_EXTERN_CRATES,
        UNUSED_FEATURES,
        UNUSED_IMPORTS,
        UNUSED_LABELS,
        UNUSED_LIFETIMES,
        UNUSED_MACRO_RULES,
        UNUSED_MACROS,
        UNUSED_MUT,
        UNUSED_QUALIFICATIONS,
        UNUSED_UNSAFE,
        UNUSED_VARIABLES,
        USELESS_DEPRECATED,
        WARNINGS,
        WASM_C_ABI,
        // tidy-alphabetical-end
    ]
}

declare_lint! {
    /// The `forbidden_lint_groups` lint detects violations of
    /// `forbid` applied to a lint group. Due to a bug in the compiler,
    /// these used to be overlooked entirely. They now generate a warning.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![forbid(warnings)]
    /// #![warn(bad_style)]
    ///
    /// fn main() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Recommended fix
    ///
    /// If your crate is using `#![forbid(warnings)]`,
    /// we recommend that you change to `#![deny(warnings)]`.
    ///
    /// ### Explanation
    ///
    /// Due to a compiler bug, applying `forbid` to lint groups
    /// previously had no effect. The bug is now fixed but instead of
    /// enforcing `forbid` we issue this future-compatibility warning
    /// to avoid breaking existing crates.
    pub FORBIDDEN_LINT_GROUPS,
    Warn,
    "applying forbid to lint-groups",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::FutureReleaseErrorReportInDeps,
        reference: "issue #81670 <https://github.com/rust-lang/rust/issues/81670>",
    };
}

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
        reason: FutureIncompatibilityReason::FutureReleaseErrorDontReportInDeps,
        reference: "issue #57571 <https://github.com/rust-lang/rust/issues/57571>",
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
        reason: FutureIncompatibilityReason::FutureReleaseErrorReportInDeps,
        reference: "issue #68585 <https://github.com/rust-lang/rust/issues/68585>",
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
    "arithmetic operation overflows",
    @eval_always = true
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
    "operation will cause a panic at runtime",
    @eval_always = true
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
    /// The `redundant_imports` lint detects imports that are redundant due to being
    /// imported already; either through a previous import, or being present in
    /// the prelude.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(redundant_imports)]
    /// use std::option::Option::None;
    /// fn foo() -> Option<i32> { None }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Redundant imports are unnecessary and can be removed to simplify code.
    /// If you intended to re-export the item to make it available outside of the
    /// module, add a visibility modifier like `pub`.
    pub REDUNDANT_IMPORTS,
    Allow,
    "imports that are redundant due to being imported already"
}

declare_lint! {
    /// The `must_not_suspend` lint guards against values that shouldn't be held across suspend points
    /// (`.await`)
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![feature(must_not_suspend)]
    /// #![warn(must_not_suspend)]
    ///
    /// #[must_not_suspend]
    /// struct SyncThing {}
    ///
    /// async fn yield_now() {}
    ///
    /// pub async fn uhoh() {
    ///     let guard = SyncThing {};
    ///     yield_now().await;
    ///     let _guard = guard;
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The `must_not_suspend` lint detects values that are marked with the `#[must_not_suspend]`
    /// attribute being held across suspend points. A "suspend" point is usually a `.await` in an async
    /// function.
    ///
    /// This attribute can be used to mark values that are semantically incorrect across suspends
    /// (like certain types of timers), values that have async alternatives, and values that
    /// regularly cause problems with the `Send`-ness of async fn's returned futures (like
    /// `MutexGuard`'s)
    ///
    pub MUST_NOT_SUSPEND,
    Allow,
    "use of a `#[must_not_suspend]` value across a yield point",
    @feature_gate = must_not_suspend;
}

declare_lint! {
    /// The `unused_extern_crates` lint guards against `extern crate` items
    /// that are never used.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(unused_extern_crates)]
    /// #![deny(warnings)]
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
    /// error: extern crate `regex` is unused in crate `lint_example`
    ///   |
    ///   = help: remove the dependency or add `use regex as _;` to the crate root
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
    ///     bar();
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
    /// The `unknown_lints` lint detects unrecognized lint attributes.
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
    "unrecognized lint attribute",
    @eval_always = true
}

declare_lint! {
    /// The `unfulfilled_lint_expectations` lint detects when a lint expectation is
    /// unfulfilled.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #[expect(unused_variables)]
    /// let x = 10;
    /// println!("{}", x);
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The `#[expect]` attribute can be used to create a lint expectation. The
    /// expectation is fulfilled, if a `#[warn]` attribute at the same location
    /// would result in a lint emission. If the expectation is unfulfilled,
    /// because no lint was emitted, this lint will be emitted on the attribute.
    ///
    pub UNFULFILLED_LINT_EXPECTATIONS,
    Warn,
    "unfulfilled lint expectation"
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
    /// consider adding a visibility modifier like `pub`.
    ///
    /// To preserve the numbering of tuple structs with unused fields,
    /// change the unused fields to have unit type or use
    /// `PhantomData`.
    ///
    /// Otherwise consider removing the unused code.
    ///
    /// ### Limitations
    ///
    /// Removing fields that are only used for side-effects and never
    /// read will result in behavioral changes. Examples of this
    /// include:
    ///
    /// - If a field's value performs an action when it is dropped.
    /// - If a field's type does not implement an auto trait
    ///   (e.g. `Send`, `Sync`, `Unpin`).
    ///
    /// For side-effects from dropping field values, this lint should
    /// be allowed on those fields. For side-effects from containing
    /// field types, `PhantomData` should be used.
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
    /// `#[allow(unused)]`) which applies to the item *following* the
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
    /// The `non_contiguous_range_endpoints` lint detects likely off-by-one errors when using
    /// exclusive [range patterns].
    ///
    /// [range patterns]: https://doc.rust-lang.org/nightly/reference/patterns.html#range-patterns
    ///
    /// ### Example
    ///
    /// ```rust
    /// let x = 123u32;
    /// match x {
    ///     0..100 => { println!("small"); }
    ///     101..1000 => { println!("large"); }
    ///     _ => { println!("larger"); }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// It is likely a mistake to have range patterns in a match expression that miss out a single
    /// number. Check that the beginning and end values are what you expect, and keep in mind that
    /// with `..=` the right bound is inclusive, and with `..` it is exclusive.
    pub NON_CONTIGUOUS_RANGE_ENDPOINTS,
    Warn,
    "detects off-by-one errors with exclusive range patterns"
}

declare_lint! {
    /// The `bindings_with_variant_name` lint detects pattern bindings with
    /// the same name as one of the matched variants.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
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
    Deny,
    "detects pattern bindings with the same name as one of the matched variants"
}

declare_lint! {
    /// The `unused_macros` lint detects macros that were not used.
    ///
    /// Note that this lint is distinct from the `unused_macro_rules` lint,
    /// which checks for single rules that never match of an otherwise used
    /// macro, and thus never expand.
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
    /// The `unused_macro_rules` lint detects macro rules that were not used.
    ///
    /// Note that the lint is distinct from the `unused_macros` lint, which
    /// fires if the entire macro is never called, while this lint fires for
    /// single unused rules of the macro that is otherwise used.
    /// `unused_macro_rules` fires only if `unused_macros` wouldn't fire.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #[warn(unused_macro_rules)]
    /// macro_rules! unused_empty {
    ///     (hello) => { println!("Hello, world!") }; // This rule is unused
    ///     () => { println!("empty") }; // This rule is used
    /// }
    ///
    /// fn main() {
    ///     unused_empty!(hello);
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Unused macro rules may signal a mistake or unfinished code. Furthermore,
    /// they slow down compilation. Right now, silencing the warning is not
    /// supported on a single rule level, so you have to add an allow to the
    /// entire macro definition.
    ///
    /// If you intended to export the macro to make it
    /// available outside of the crate, use the [`macro_export` attribute].
    ///
    /// [`macro_export` attribute]: https://doc.rust-lang.org/reference/macros-by-example.html#path-based-scope
    pub UNUSED_MACRO_RULES,
    Allow,
    "detects macro rules that were not used"
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
    /// with coercion, which may require a temporary variable.
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
    /// warning in the future, possibly with an explicit syntax for coercions
    /// providing a convenient way to work around the current issues.
    /// See [RFC 401 (coercions)][rfc-401], [RFC 803 (type ascription)][rfc-803] and
    /// [RFC 3307 (remove type ascription)][rfc-3307] for historical context.
    ///
    /// [rfc-401]: https://github.com/rust-lang/rfcs/blob/master/text/0401-coercions.md
    /// [rfc-803]: https://github.com/rust-lang/rfcs/blob/master/text/0803-type-ascription.md
    /// [rfc-3307]: https://github.com/rust-lang/rfcs/blob/master/text/3307-de-rfc-type-ascription.md
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
    /// warning in the future, possibly with an explicit syntax for coercions
    /// providing a convenient way to work around the current issues.
    /// See [RFC 401 (coercions)][rfc-401], [RFC 803 (type ascription)][rfc-803] and
    /// [RFC 3307 (remove type ascription)][rfc-3307] for historical context.
    ///
    /// [rfc-401]: https://github.com/rust-lang/rfcs/blob/master/text/0401-coercions.md
    /// [rfc-803]: https://github.com/rust-lang/rfcs/blob/master/text/0803-type-ascription.md
    /// [rfc-3307]: https://github.com/rust-lang/rfcs/blob/master/text/3307-de-rfc-type-ascription.md
    pub TRIVIAL_NUMERIC_CASTS,
    Allow,
    "detects trivial casts of numeric types which could be removed"
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
    /// #127909] for more details.
    ///
    /// [issue #127909]: https://github.com/rust-lang/rust/issues/127909
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub PUB_USE_OF_PRIVATE_EXTERN_CRATE,
    Deny,
    "detect public re-exports of private extern crates",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::FutureReleaseErrorReportInDeps,
        reference: "issue #127909 <https://github.com/rust-lang/rust/issues/127909>",
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
        reason: FutureIncompatibilityReason::FutureReleaseErrorReportInDeps,
        reference: "issue #36887 <https://github.com/rust-lang/rust/issues/36887>",
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
    ///     * For more complex single-initialization cases, consider using
    ///       [`std::sync::LazyLock`].
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
    /// [`std::sync::LazyLock`]: https://doc.rust-lang.org/stable/std/sync/struct.LazyLock.html
    /// [`atomic`]: https://doc.rust-lang.org/std/sync/atomic/index.html
    /// [`Mutex`]: https://doc.rust-lang.org/std/sync/struct.Mutex.html
    pub CONST_ITEM_MUTATION,
    Warn,
    "detects attempts to mutate a `const` item",
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
        reason: FutureIncompatibilityReason::FutureReleaseErrorDontReportInDeps,
        reference: "issue #35203 <https://github.com/rust-lang/rust/issues/35203>",
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
    /// ```rust,compile_fail,edition2021
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
        reason: FutureIncompatibilityReason::FutureReleaseErrorReportInDeps,
        reference: "issue #40107 <https://github.com/rust-lang/rust/issues/40107>",
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
    ///     fn late(self, _: &u8, _: &u8) {}
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
        reason: FutureIncompatibilityReason::FutureReleaseErrorDontReportInDeps,
        reference: "issue #42868 <https://github.com/rust-lang/rust/issues/42868>",
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
        reason: FutureIncompatibilityReason::Custom("the behavior may change in a future release"),
        reference: "issue #56105 <https://github.com/rust-lang/rust/issues/56105>",
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
    /// The `rust_2024_incompatible_pat` lint
    /// detects patterns whose meaning will change in the Rust 2024 edition.
    ///
    /// ### Example
    ///
    /// ```rust,edition2021
    /// #![warn(rust_2024_incompatible_pat)]
    ///
    /// if let Some(&a) = &Some(&0u8) {
    ///     let _: u8 = a;
    /// }
    /// if let Some(mut _a) = &mut Some(0u8) {
    ///     _a = 7u8;
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// In Rust 2024 and above, the `mut` keyword does not reset the pattern binding mode,
    /// and nor do `&` or `&mut` patterns. The lint will suggest code that
    /// has the same meaning in all editions.
    pub RUST_2024_INCOMPATIBLE_PAT,
    Allow,
    "detects patterns whose meaning will change in Rust 2024",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionSemanticsChange(Edition::Edition2024),
        reference: "<https://doc.rust-lang.org/nightly/edition-guide/rust-2024/match-ergonomics.html>",
    };
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
    /// The `redundant_lifetimes` lint detects lifetime parameters that are
    /// redundant because they are equal to another named lifetime.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #[deny(redundant_lifetimes)]
    ///
    /// // `'a = 'static`, so all usages of `'a` can be replaced with `'static`
    /// pub fn bar<'a: 'static>() {}
    ///
    /// // `'a = 'b`, so all usages of `'b` can be replaced with `'a`
    /// pub fn bar<'a: 'b, 'b: 'a>() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Unused lifetime parameters may signal a mistake or unfinished code.
    /// Consider removing the parameter.
    pub REDUNDANT_LIFETIMES,
    Allow,
    "detects lifetime parameters that are redundant because they are equal to some other named lifetime"
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
        reason: FutureIncompatibilityReason::EditionError(Edition::Edition2018),
        reference: "issue #46906 <https://github.com/rust-lang/rust/issues/46906>",
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
    /// #![deny(warnings)]
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
    "hidden lifetime parameters in types are deprecated"
}

declare_lint! {
    /// The `elided_named_lifetimes` lint detects when an elided
    /// lifetime ends up being a named lifetime, such as `'static`
    /// or some lifetime parameter `'a`.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(elided_named_lifetimes)]
    /// struct Foo;
    /// impl Foo {
    ///     pub fn get_mut(&'static self, x: &mut u8) -> &mut u8 {
    ///         unsafe { &mut *(x as *mut _) }
    ///     }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Lifetime elision is quite useful, because it frees you from having
    /// to give each lifetime its own name, but sometimes it can produce
    /// somewhat surprising resolutions. In safe code, it is mostly okay,
    /// because the borrow checker prevents any unsoundness, so the worst
    /// case scenario is you get a confusing error message in some other place.
    /// But with `unsafe` code, such unexpected resolutions may lead to unsound code.
    pub ELIDED_NAMED_LIFETIMES,
    Warn,
    "detects when an elided lifetime gets resolved to be `'static` or some named parameter"
}

declare_lint! {
    /// The `bare_trait_objects` lint suggests using `dyn Trait` for trait
    /// objects.
    ///
    /// ### Example
    ///
    /// ```rust,edition2018
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
    "suggest using `dyn Trait` for trait objects",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionError(Edition::Edition2021),
        reference: "<https://doc.rust-lang.org/nightly/edition-guide/rust-2021/warnings-promoted-to-error.html>",
    };
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
        reason: FutureIncompatibilityReason::EditionError(Edition::Edition2018),
        reference: "issue #53130 <https://github.com/rust-lang/rust/issues/53130>",
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
    ///     // is_partitioned is an unstable method that already exists on the Iterator trait
    ///     fn is_partitioned<P>(self, predicate: P) -> bool
    ///     where
    ///         Self: Sized,
    ///         P: FnMut(Self::Item) -> bool,
    ///     {true}
    /// }
    ///
    /// impl<T: ?Sized> MyIterator for T where T: Iterator { }
    ///
    /// let x = vec![1, 2, 3];
    /// let _ = x.iter().is_partitioned(|_| true);
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// When new methods are added to traits in the standard library, they are
    /// usually added in an "unstable" form which is only available on the
    /// [nightly channel] with a [`feature` attribute]. If there is any
    /// preexisting code which extends a trait to have a method with the same
    /// name, then the names will collide. In the future, when the method is
    /// stabilized, this will cause an error due to the ambiguity. This lint
    /// is an early-warning to let you know that there may be a collision in
    /// the future. This can be avoided by adding type annotations to
    /// disambiguate which trait method you intend to call, such as
    /// `MyIterator::is_partitioned(my_iter, my_predicate)` or renaming or removing the method.
    ///
    /// [nightly channel]: https://doc.rust-lang.org/book/appendix-07-nightly-rust.html
    /// [`feature` attribute]: https://doc.rust-lang.org/nightly/unstable-book/
    pub UNSTABLE_NAME_COLLISIONS,
    Warn,
    "detects name collision with an existing but unstable method",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::Custom(
            "once this associated item is added to the standard library, \
             the ambiguity may cause an error or change in behavior!"
        ),
        reference: "issue #48919 <https://github.com/rust-lang/rust/issues/48919>",
        // Note: this item represents future incompatibility of all unstable functions in the
        //       standard library, and thus should never be removed or changed to an error.
    };
}

declare_lint! {
    /// The `irrefutable_let_patterns` lint detects [irrefutable patterns]
    /// in [`if let`]s, [`while let`]s, and `if let` guards.
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
    /// `if let` or `while let` statement, because the pattern will always match
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
    /// [`if let`]: https://doc.rust-lang.org/reference/expressions/if-expr.html#if-let-expressions
    /// [`while let`]: https://doc.rust-lang.org/reference/expressions/loop-expr.html#predicate-pattern-loops
    /// [`let`]: https://doc.rust-lang.org/reference/statements.html#let-statements
    /// [`loop`]: https://doc.rust-lang.org/reference/expressions/loop-expr.html#infinite-loops
    /// [RFC 2086]: https://github.com/rust-lang/rfcs/blob/master/text/2086-allow-if-let-irrefutables.md
    pub IRREFUTABLE_LET_PATTERNS,
    Warn,
    "detects irrefutable patterns in `if let` and `while let` statements"
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
    Deny,
    "detects proc macro derives using inaccessible names from parent modules",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::FutureReleaseErrorReportInDeps,
        reference: "issue #83583 <https://github.com/rust-lang/rust/issues/83583>",
    };
}

declare_lint! {
    /// The `macro_use_extern_crate` lint detects the use of the [`macro_use` attribute].
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
    /// error: applying the `#[macro_use]` attribute to an `extern crate` item is deprecated
    ///  --> src/main.rs:3:1
    ///   |
    /// 3 | #[macro_use]
    ///   | ^^^^^^^^^^^^
    ///   |
    ///   = help: remove it and import macros at use sites with a `use` item instead
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
        reason: FutureIncompatibilityReason::FutureReleaseErrorDontReportInDeps,
        reference: "issue #52234 <https://github.com/rust-lang/rust/issues/52234>",
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
    /// #![deny(warnings)]
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
    /// The `deprecated_in_future` lint is internal to rustc and should not be
    /// used by user code.
    ///
    /// This lint is only enabled in the standard library. It works with the
    /// use of `#[deprecated]` with a `since` field of a version in the future.
    /// This allows something to be marked as deprecated in a future version,
    /// and then this lint will ensure that the item is no longer used in the
    /// standard library. See the [stability documentation] for more details.
    ///
    /// [stability documentation]: https://rustc-dev-guide.rust-lang.org/stability.html#deprecated
    pub DEPRECATED_IN_FUTURE,
    Allow,
    "detects use of items that will be deprecated in a future version",
    report_in_external_macro
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
        reason: FutureIncompatibilityReason::FutureReleaseErrorDontReportInDeps,
        reference: "issue #57644 <https://github.com/rust-lang/rust/issues/57644>",
    };
}

declare_lint! {
    /// The `soft_unstable` lint detects unstable features that were unintentionally allowed on
    /// stable. This is a [future-incompatible] lint to transition this to a hard error in the
    /// future. See [issue #64266] for more details.
    ///
    /// [issue #64266]: https://github.com/rust-lang/rust/issues/64266
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub SOFT_UNSTABLE,
    Deny,
    "a feature gate that doesn't break dependent crates",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::FutureReleaseErrorReportInDeps,
        reference: "issue #64266 <https://github.com/rust-lang/rust/issues/64266>",
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
    /// ```rust,ignore (fails on non-x86_64)
    /// #[cfg(target_arch="x86_64")]
    /// use std::arch::asm;
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
    ///  --> src/main.rs:7:19
    ///   |
    /// 7 |         asm!("mov {0}, {0}", in(reg) 0i16);
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
    /// See [register template modifiers] in the reference for more details.
    ///
    /// [register template modifiers]: https://doc.rust-lang.org/nightly/reference/inline-assembly.html#template-modifiers
    pub ASM_SUB_REGISTER,
    Warn,
    "using only a subset of a register for inline asm inputs",
}

declare_lint! {
    /// The `bad_asm_style` lint detects the use of the `.intel_syntax` and
    /// `.att_syntax` directives.
    ///
    /// ### Example
    ///
    /// ```rust,ignore (fails on non-x86_64)
    /// #[cfg(target_arch="x86_64")]
    /// use std::arch::asm;
    ///
    /// fn main() {
    ///     #[cfg(target_arch="x86_64")]
    ///     unsafe {
    ///         asm!(
    ///             ".att_syntax",
    ///             "movq %{0}, %{0}", in(reg) 0usize
    ///         );
    ///     }
    /// }
    /// ```
    ///
    /// This will produce:
    ///
    /// ```text
    /// warning: avoid using `.att_syntax`, prefer using `options(att_syntax)` instead
    ///  --> src/main.rs:8:14
    ///   |
    /// 8 |             ".att_syntax",
    ///   |              ^^^^^^^^^^^
    ///   |
    ///   = note: `#[warn(bad_asm_style)]` on by default
    /// ```
    ///
    /// ### Explanation
    ///
    /// On x86, `asm!` uses the intel assembly syntax by default. While this
    /// can be switched using assembler directives like `.att_syntax`, using the
    /// `att_syntax` option is recommended instead because it will also properly
    /// prefix register placeholders with `%` as required by AT&T syntax.
    pub BAD_ASM_STYLE,
    Warn,
    "incorrect use of inline assembly",
}

declare_lint! {
    /// The `unsafe_op_in_unsafe_fn` lint detects unsafe operations in unsafe
    /// functions without an explicit unsafe block.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
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
    /// This lint is "allow" by default on editions up to 2021, from 2024 it is
    /// "warn" by default; the plan for increasing severity further is
    /// still being considered. See [RFC #2585] and [issue #71668] for more
    /// details.
    ///
    /// [`unsafe fn`]: https://doc.rust-lang.org/reference/unsafe-functions.html
    /// [`unsafe` block]: https://doc.rust-lang.org/reference/expressions/block-expr.html#unsafe-blocks
    /// [unsafe]: https://doc.rust-lang.org/reference/unsafety.html
    /// [RFC #2585]: https://github.com/rust-lang/rfcs/blob/master/text/2585-unsafe-block-in-unsafe-fn.md
    /// [issue #71668]: https://github.com/rust-lang/rust/issues/71668
    pub UNSAFE_OP_IN_UNSAFE_FN,
    Allow,
    "unsafe operations in unsafe functions without an explicit unsafe block are deprecated",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionSemanticsChange(Edition::Edition2024),
        reference: "<https://doc.rust-lang.org/nightly/edition-guide/rust-2024/unsafe-op-in-unsafe-fn.html>",
        explain_reason: false
    };
    @edition Edition2024 => Warn;
}

declare_lint! {
    /// The `fuzzy_provenance_casts` lint detects an `as` cast between an integer
    /// and a pointer.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![feature(strict_provenance_lints)]
    /// #![warn(fuzzy_provenance_casts)]
    ///
    /// fn main() {
    ///     let _dangling = 16_usize as *const u8;
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// This lint is part of the strict provenance effort, see [issue #95228].
    /// Casting an integer to a pointer is considered bad style, as a pointer
    /// contains, besides the *address* also a *provenance*, indicating what
    /// memory the pointer is allowed to read/write. Casting an integer, which
    /// doesn't have provenance, to a pointer requires the compiler to assign
    /// (guess) provenance. The compiler assigns "all exposed valid" (see the
    /// docs of [`ptr::with_exposed_provenance`] for more information about this
    /// "exposing"). This penalizes the optimiser and is not well suited for
    /// dynamic analysis/dynamic program verification (e.g. Miri or CHERI
    /// platforms).
    ///
    /// It is much better to use [`ptr::with_addr`] instead to specify the
    /// provenance you want. If using this function is not possible because the
    /// code relies on exposed provenance then there is as an escape hatch
    /// [`ptr::with_exposed_provenance`].
    ///
    /// [issue #95228]: https://github.com/rust-lang/rust/issues/95228
    /// [`ptr::with_addr`]: https://doc.rust-lang.org/core/primitive.pointer.html#method.with_addr
    /// [`ptr::with_exposed_provenance`]: https://doc.rust-lang.org/core/ptr/fn.with_exposed_provenance.html
    pub FUZZY_PROVENANCE_CASTS,
    Allow,
    "a fuzzy integer to pointer cast is used",
    @feature_gate = strict_provenance_lints;
}

declare_lint! {
    /// The `lossy_provenance_casts` lint detects an `as` cast between a pointer
    /// and an integer.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![feature(strict_provenance_lints)]
    /// #![warn(lossy_provenance_casts)]
    ///
    /// fn main() {
    ///     let x: u8 = 37;
    ///     let _addr: usize = &x as *const u8 as usize;
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// This lint is part of the strict provenance effort, see [issue #95228].
    /// Casting a pointer to an integer is a lossy operation, because beyond
    /// just an *address* a pointer may be associated with a particular
    /// *provenance*. This information is used by the optimiser and for dynamic
    /// analysis/dynamic program verification (e.g. Miri or CHERI platforms).
    ///
    /// Since this cast is lossy, it is considered good style to use the
    /// [`ptr::addr`] method instead, which has a similar effect, but doesn't
    /// "expose" the pointer provenance. This improves optimisation potential.
    /// See the docs of [`ptr::addr`] and [`ptr::expose_provenance`] for more information
    /// about exposing pointer provenance.
    ///
    /// If your code can't comply with strict provenance and needs to expose
    /// the provenance, then there is [`ptr::expose_provenance`] as an escape hatch,
    /// which preserves the behaviour of `as usize` casts while being explicit
    /// about the semantics.
    ///
    /// [issue #95228]: https://github.com/rust-lang/rust/issues/95228
    /// [`ptr::addr`]: https://doc.rust-lang.org/core/primitive.pointer.html#method.addr
    /// [`ptr::expose_provenance`]: https://doc.rust-lang.org/core/primitive.pointer.html#method.expose_provenance
    pub LOSSY_PROVENANCE_CASTS,
    Allow,
    "a lossy pointer to integer cast is used",
    @feature_gate = strict_provenance_lints;
}

declare_lint! {
    /// The `const_evaluatable_unchecked` lint detects a generic constant used
    /// in a type.
    ///
    /// ### Example
    ///
    /// ```rust
    /// const fn foo<T>() -> usize {
    ///     if size_of::<*mut T>() < 8 { // size of *mut T does not depend on T
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
        reason: FutureIncompatibilityReason::FutureReleaseErrorDontReportInDeps,
        reference: "issue #76200 <https://github.com/rust-lang/rust/issues/76200>",
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
    /// unsafe extern {
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
    /// statics). This was accidentally allowed, but is being phased out.
    pub UNINHABITED_STATIC,
    Warn,
    "uninhabited static",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::FutureReleaseErrorDontReportInDeps,
        reference: "issue #74840 <https://github.com/rust-lang/rust/issues/74840>",
    };
}

declare_lint! {
    /// The `unnameable_test_items` lint detects [`#[test]`][test] functions
    /// that are not able to be run by the test harness because they are in a
    /// position where they are not nameable.
    ///
    /// [test]: https://doc.rust-lang.org/reference/attributes/testing.html#the-test-attribute
    ///
    /// ### Example
    ///
    /// ```rust,test
    /// fn main() {
    ///     #[test]
    ///     fn foo() {
    ///         // This test will not fail because it does not run.
    ///         assert_eq!(1, 2);
    ///     }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// In order for the test harness to run a test, the test function must be
    /// located in a position where it can be accessed from the crate root.
    /// This generally means it must be defined in a module, and not anywhere
    /// else such as inside another function. The compiler previously allowed
    /// this without an error, so a lint was added as an alert that a test is
    /// not being used. Whether or not this should be allowed has not yet been
    /// decided, see [RFC 2471] and [issue #36629].
    ///
    /// [RFC 2471]: https://github.com/rust-lang/rfcs/pull/2471#issuecomment-397414443
    /// [issue #36629]: https://github.com/rust-lang/rust/issues/36629
    pub UNNAMEABLE_TEST_ITEMS,
    Warn,
    "detects an item that cannot be named being marked as `#[test_case]`",
    report_in_external_macro
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
    /// The `ineffective_unstable_trait_impl` lint detects `#[unstable]` attributes which are not used.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![feature(staged_api)]
    ///
    /// #[derive(Clone)]
    /// #[stable(feature = "x", since = "1")]
    /// struct S {}
    ///
    /// #[unstable(feature = "y", issue = "none")]
    /// impl Copy for S {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// `staged_api` does not currently support using a stability attribute on `impl` blocks.
    /// `impl`s are always stable if both the type and trait are stable, and always unstable otherwise.
    pub INEFFECTIVE_UNSTABLE_TRAIT_IMPL,
    Deny,
    "detects `#[unstable]` on stable trait implementations for stable types"
}

declare_lint! {
    /// The `self_constructor_from_outer_item` lint detects cases where the `Self` constructor
    /// was silently allowed due to a bug in the resolver, and which may produce surprising
    /// and unintended behavior.
    ///
    /// Using a `Self` type alias from an outer item was never intended, but was silently allowed.
    /// This is deprecated -- and is a hard error when the `Self` type alias references generics
    /// that are not in scope.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(self_constructor_from_outer_item)]
    ///
    /// struct S0(usize);
    ///
    /// impl S0 {
    ///     fn foo() {
    ///         const C: S0 = Self(0);
    ///         fn bar() -> S0 {
    ///             Self(0)
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The `Self` type alias should not be reachable because nested items are not associated with
    /// the scope of the parameters from the parent item.
    pub SELF_CONSTRUCTOR_FROM_OUTER_ITEM,
    Warn,
    "detect unsupported use of `Self` from outer item",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::FutureReleaseErrorDontReportInDeps,
        reference: "issue #124186 <https://github.com/rust-lang/rust/issues/124186>",
    };
}

declare_lint! {
    /// The `semicolon_in_expressions_from_macros` lint detects trailing semicolons
    /// in macro bodies when the macro is invoked in expression position.
    /// This was previous accepted, but is being phased out.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(semicolon_in_expressions_from_macros)]
    /// macro_rules! foo {
    ///     () => { true; }
    /// }
    ///
    /// fn main() {
    ///     let val = match true {
    ///         true => false,
    ///         _ => foo!()
    ///     };
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Previous, Rust ignored trailing semicolon in a macro
    /// body when a macro was invoked in expression position.
    /// However, this makes the treatment of semicolons in the language
    /// inconsistent, and could lead to unexpected runtime behavior
    /// in some circumstances (e.g. if the macro author expects
    /// a value to be dropped).
    ///
    /// This is a [future-incompatible] lint to transition this
    /// to a hard error in the future. See [issue #79813] for more details.
    ///
    /// [issue #79813]: https://github.com/rust-lang/rust/issues/79813
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub SEMICOLON_IN_EXPRESSIONS_FROM_MACROS,
    Warn,
    "trailing semicolon in macro body used as expression",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::FutureReleaseErrorReportInDeps,
        reference: "issue #79813 <https://github.com/rust-lang/rust/issues/79813>",
    };
}

declare_lint! {
    /// The `legacy_derive_helpers` lint detects derive helper attributes
    /// that are used before they are introduced.
    ///
    /// ### Example
    ///
    /// ```rust,ignore (needs extern crate)
    /// #[serde(rename_all = "camelCase")]
    /// #[derive(Deserialize)]
    /// struct S { /* fields */ }
    /// ```
    ///
    /// produces:
    ///
    /// ```text
    /// warning: derive helper attribute is used before it is introduced
    ///   --> $DIR/legacy-derive-helpers.rs:1:3
    ///    |
    ///  1 | #[serde(rename_all = "camelCase")]
    ///    |   ^^^^^
    /// ...
    ///  2 | #[derive(Deserialize)]
    ///    |          ----------- the attribute is introduced here
    /// ```
    ///
    /// ### Explanation
    ///
    /// Attributes like this work for historical reasons, but attribute expansion works in
    /// left-to-right order in general, so, to resolve `#[serde]`, compiler has to try to "look
    /// into the future" at not yet expanded part of the item , but such attempts are not always
    /// reliable.
    ///
    /// To fix the warning place the helper attribute after its corresponding derive.
    /// ```rust,ignore (needs extern crate)
    /// #[derive(Deserialize)]
    /// #[serde(rename_all = "camelCase")]
    /// struct S { /* fields */ }
    /// ```
    pub LEGACY_DERIVE_HELPERS,
    Warn,
    "detects derive helper attributes that are used before they are introduced",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::FutureReleaseErrorDontReportInDeps,
        reference: "issue #79202 <https://github.com/rust-lang/rust/issues/79202>",
    };
}

declare_lint! {
    /// The `large_assignments` lint detects when objects of large
    /// types are being moved around.
    ///
    /// ### Example
    ///
    /// ```rust,ignore (can crash on some platforms)
    /// let x = [0; 50000];
    /// let y = x;
    /// ```
    ///
    /// produces:
    ///
    /// ```text
    /// warning: moving a large value
    ///   --> $DIR/move-large.rs:1:3
    ///   let y = x;
    ///           - Copied large value here
    /// ```
    ///
    /// ### Explanation
    ///
    /// When using a large type in a plain assignment or in a function
    /// argument, idiomatic code can be inefficient.
    /// Ideally appropriate optimizations would resolve this, but such
    /// optimizations are only done in a best-effort manner.
    /// This lint will trigger on all sites of large moves and thus allow the
    /// user to resolve them in code.
    pub LARGE_ASSIGNMENTS,
    Warn,
    "detects large moves or copies",
}

declare_lint! {
    /// The `unexpected_cfgs` lint detects unexpected conditional compilation conditions.
    ///
    /// ### Example
    ///
    /// ```text
    /// rustc --check-cfg 'cfg()'
    /// ```
    ///
    /// ```rust,ignore (needs command line option)
    /// #[cfg(widnows)]
    /// fn foo() {}
    /// ```
    ///
    /// This will produce:
    ///
    /// ```text
    /// warning: unexpected `cfg` condition name: `widnows`
    ///  --> lint_example.rs:1:7
    ///   |
    /// 1 | #[cfg(widnows)]
    ///   |       ^^^^^^^
    ///   |
    ///   = note: `#[warn(unexpected_cfgs)]` on by default
    /// ```
    ///
    /// ### Explanation
    ///
    /// This lint is only active when [`--check-cfg`][check-cfg] arguments are being
    /// passed to the compiler and triggers whenever an unexpected condition name or value is
    /// used.
    ///
    /// See the [Checking Conditional Configurations][check-cfg] section for more
    /// details.
    ///
    /// See the [Cargo Specifics][unexpected_cfgs_lint_config] section for configuring this lint in
    /// `Cargo.toml`.
    ///
    /// [check-cfg]: https://doc.rust-lang.org/nightly/rustc/check-cfg.html
    /// [unexpected_cfgs_lint_config]: https://doc.rust-lang.org/nightly/rustc/check-cfg/cargo-specifics.html#check-cfg-in-lintsrust-table
    pub UNEXPECTED_CFGS,
    Warn,
    "detects unexpected names and values in `#[cfg]` conditions",
    report_in_external_macro
}

declare_lint! {
    /// The `explicit_builtin_cfgs_in_flags` lint detects builtin cfgs set via the `--cfg` flag.
    ///
    /// ### Example
    ///
    /// ```text
    /// rustc --cfg unix
    /// ```
    ///
    /// ```rust,ignore (needs command line option)
    /// fn main() {}
    /// ```
    ///
    /// This will produce:
    ///
    /// ```text
    /// error: unexpected `--cfg unix` flag
    ///   |
    ///   = note: config `unix` is only supposed to be controlled by `--target`
    ///   = note: manually setting a built-in cfg can and does create incoherent behaviors
    ///   = note: `#[deny(explicit_builtin_cfgs_in_flags)]` on by default
    /// ```
    ///
    /// ### Explanation
    ///
    /// Setting builtin cfgs can and does produce incoherent behavior, it's better to the use
    /// the appropriate `rustc` flag that controls the config. For example setting the `windows`
    /// cfg but on Linux based target.
    pub EXPLICIT_BUILTIN_CFGS_IN_FLAGS,
    Deny,
    "detects builtin cfgs set via the `--cfg`"
}

declare_lint! {
    /// The `repr_transparent_external_private_fields` lint
    /// detects types marked `#[repr(transparent)]` that (transitively)
    /// contain an external ZST type marked `#[non_exhaustive]` or containing
    /// private fields
    ///
    /// ### Example
    ///
    /// ```rust,ignore (needs external crate)
    /// #![deny(repr_transparent_external_private_fields)]
    /// use foo::NonExhaustiveZst;
    ///
    /// #[repr(transparent)]
    /// struct Bar(u32, ([u32; 0], NonExhaustiveZst));
    /// ```
    ///
    /// This will produce:
    ///
    /// ```text
    /// error: zero-sized fields in repr(transparent) cannot contain external non-exhaustive types
    ///  --> src/main.rs:5:28
    ///   |
    /// 5 | struct Bar(u32, ([u32; 0], NonExhaustiveZst));
    ///   |                            ^^^^^^^^^^^^^^^^
    ///   |
    /// note: the lint level is defined here
    ///  --> src/main.rs:1:9
    ///   |
    /// 1 | #![deny(repr_transparent_external_private_fields)]
    ///   |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ///   = warning: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    ///   = note: for more information, see issue #78586 <https://github.com/rust-lang/rust/issues/78586>
    ///   = note: this struct contains `NonExhaustiveZst`, which is marked with `#[non_exhaustive]`, and makes it not a breaking change to become non-zero-sized in the future.
    /// ```
    ///
    /// ### Explanation
    ///
    /// Previous, Rust accepted fields that contain external private zero-sized types,
    /// even though it should not be a breaking change to add a non-zero-sized field to
    /// that private type.
    ///
    /// This is a [future-incompatible] lint to transition this
    /// to a hard error in the future. See [issue #78586] for more details.
    ///
    /// [issue #78586]: https://github.com/rust-lang/rust/issues/78586
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub REPR_TRANSPARENT_EXTERNAL_PRIVATE_FIELDS,
    Warn,
    "transparent type contains an external ZST that is marked #[non_exhaustive] or contains private fields",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::FutureReleaseErrorDontReportInDeps,
        reference: "issue #78586 <https://github.com/rust-lang/rust/issues/78586>",
    };
}

declare_lint! {
    /// The `unstable_syntax_pre_expansion` lint detects the use of unstable
    /// syntax that is discarded during attribute expansion.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #[cfg(FALSE)]
    /// macro foo() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The input to active attributes such as `#[cfg]` or procedural macro
    /// attributes is required to be valid syntax. Previously, the compiler only
    /// gated the use of unstable syntax features after resolving `#[cfg]` gates
    /// and expanding procedural macros.
    ///
    /// To avoid relying on unstable syntax, move the use of unstable syntax
    /// into a position where the compiler does not parse the syntax, such as a
    /// functionlike macro.
    ///
    /// ```rust
    /// # #![deny(unstable_syntax_pre_expansion)]
    ///
    /// macro_rules! identity {
    ///    ( $($tokens:tt)* ) => { $($tokens)* }
    /// }
    ///
    /// #[cfg(FALSE)]
    /// identity! {
    ///    macro foo() {}
    /// }
    /// ```
    ///
    /// This is a [future-incompatible] lint to transition this
    /// to a hard error in the future. See [issue #65860] for more details.
    ///
    /// [issue #65860]: https://github.com/rust-lang/rust/issues/65860
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub UNSTABLE_SYNTAX_PRE_EXPANSION,
    Warn,
    "unstable syntax can change at any point in the future, causing a hard error!",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::FutureReleaseErrorDontReportInDeps,
        reference: "issue #65860 <https://github.com/rust-lang/rust/issues/65860>",
    };
}

declare_lint! {
    /// The `ambiguous_glob_reexports` lint detects cases where names re-exported via globs
    /// collide. Downstream users trying to use the same name re-exported from multiple globs
    /// will receive a warning pointing out redefinition of the same name.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(ambiguous_glob_reexports)]
    /// pub mod foo {
    ///     pub type X = u8;
    /// }
    ///
    /// pub mod bar {
    ///     pub type Y = u8;
    ///     pub type X = u8;
    /// }
    ///
    /// pub use foo::*;
    /// pub use bar::*;
    ///
    ///
    /// pub fn main() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// This was previously accepted but it could silently break a crate's downstream users code.
    /// For example, if `foo::*` and `bar::*` were re-exported before `bar::X` was added to the
    /// re-exports, down stream users could use `this_crate::X` without problems. However, adding
    /// `bar::X` would cause compilation errors in downstream crates because `X` is defined
    /// multiple times in the same namespace of `this_crate`.
    pub AMBIGUOUS_GLOB_REEXPORTS,
    Warn,
    "ambiguous glob re-exports",
}

declare_lint! {
    /// The `hidden_glob_reexports` lint detects cases where glob re-export items are shadowed by
    /// private items.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(hidden_glob_reexports)]
    ///
    /// pub mod upstream {
    ///     mod inner { pub struct Foo {}; pub struct Bar {}; }
    ///     pub use self::inner::*;
    ///     struct Foo {} // private item shadows `inner::Foo`
    /// }
    ///
    /// // mod downstream {
    /// //     fn test() {
    /// //         let _ = crate::upstream::Foo; // inaccessible
    /// //     }
    /// // }
    ///
    /// pub fn main() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// This was previously accepted without any errors or warnings but it could silently break a
    /// crate's downstream user code. If the `struct Foo` was added, `dep::inner::Foo` would
    /// silently become inaccessible and trigger a "`struct `Foo` is private`" visibility error at
    /// the downstream use site.
    pub HIDDEN_GLOB_REEXPORTS,
    Warn,
    "name introduced by a private item shadows a name introduced by a public glob re-export",
}

declare_lint! {
    /// The `long_running_const_eval` lint is emitted when const
    /// eval is running for a long time to ensure rustc terminates
    /// even if you accidentally wrote an infinite loop.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// const FOO: () = loop {};
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Loops allow const evaluation to compute arbitrary code, but may also
    /// cause infinite loops or just very long running computations.
    /// Users can enable long running computations by allowing the lint
    /// on individual constants or for entire crates.
    ///
    /// ### Unconditional warnings
    ///
    /// Note that regardless of whether the lint is allowed or set to warn,
    /// the compiler will issue warnings if constant evaluation runs significantly
    /// longer than this lint's limit. These warnings are also shown to downstream
    /// users from crates.io or similar registries. If you are above the lint's limit,
    /// both you and downstream users might be exposed to these warnings.
    /// They might also appear on compiler updates, as the compiler makes minor changes
    /// about how complexity is measured: staying below the limit ensures that there
    /// is enough room, and given that the lint is disabled for people who use your
    /// dependency it means you will be the only one to get the warning and can put
    /// out an update in your own time.
    pub LONG_RUNNING_CONST_EVAL,
    Deny,
    "detects long const eval operations",
    report_in_external_macro
}

declare_lint! {
    /// The `unused_associated_type_bounds` lint is emitted when an
    /// associated type bound is added to a trait object, but the associated
    /// type has a `where Self: Sized` bound, and is thus unavailable on the
    /// trait object anyway.
    ///
    /// ### Example
    ///
    /// ```rust
    /// trait Foo {
    ///     type Bar where Self: Sized;
    /// }
    /// type Mop = dyn Foo<Bar = ()>;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Just like methods with `Self: Sized` bounds are unavailable on trait
    /// objects, associated types can be removed from the trait object.
    pub UNUSED_ASSOCIATED_TYPE_BOUNDS,
    Warn,
    "detects unused `Foo = Bar` bounds in `dyn Trait<Foo = Bar>`"
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

declare_lint! {
    /// The `rust_2021_incompatible_closure_captures` lint detects variables that aren't completely
    /// captured in Rust 2021, such that the `Drop` order of their fields may differ between
    /// Rust 2018 and 2021.
    ///
    /// It can also detect when a variable implements a trait like `Send`, but one of its fields does not,
    /// and the field is captured by a closure and used with the assumption that said field implements
    /// the same trait as the root variable.
    ///
    /// ### Example of drop reorder
    ///
    /// ```rust,edition2018,compile_fail
    /// #![deny(rust_2021_incompatible_closure_captures)]
    /// # #![allow(unused)]
    ///
    /// struct FancyInteger(i32);
    ///
    /// impl Drop for FancyInteger {
    ///     fn drop(&mut self) {
    ///         println!("Just dropped {}", self.0);
    ///     }
    /// }
    ///
    /// struct Point { x: FancyInteger, y: FancyInteger }
    ///
    /// fn main() {
    ///   let p = Point { x: FancyInteger(10), y: FancyInteger(20) };
    ///
    ///   let c = || {
    ///      let x = p.x;
    ///   };
    ///
    ///   c();
    ///
    ///   // ... More code ...
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// In the above example, `p.y` will be dropped at the end of `f` instead of
    /// with `c` in Rust 2021.
    ///
    /// ### Example of auto-trait
    ///
    /// ```rust,edition2018,compile_fail
    /// #![deny(rust_2021_incompatible_closure_captures)]
    /// use std::thread;
    ///
    /// struct Pointer(*mut i32);
    /// unsafe impl Send for Pointer {}
    ///
    /// fn main() {
    ///     let mut f = 10;
    ///     let fptr = Pointer(&mut f as *mut i32);
    ///     thread::spawn(move || unsafe {
    ///         *fptr.0 = 20;
    ///     });
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// In the above example, only `fptr.0` is captured in Rust 2021.
    /// The field is of type `*mut i32`, which doesn't implement `Send`,
    /// making the code invalid as the field cannot be sent between threads safely.
    pub RUST_2021_INCOMPATIBLE_CLOSURE_CAPTURES,
    Allow,
    "detects closures affected by Rust 2021 changes",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionSemanticsChange(Edition::Edition2021),
        explain_reason: false,
    };
}

declare_lint_pass!(UnusedDocComment => [UNUSED_DOC_COMMENTS]);

declare_lint! {
    /// The `missing_abi` lint detects cases where the ABI is omitted from
    /// `extern` declarations.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(missing_abi)]
    ///
    /// extern fn foo() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// For historic reasons, Rust implicitly selects `C` as the default ABI for
    /// `extern` declarations. [Other ABIs] like `C-unwind` and `system` have
    /// been added since then, and especially with their addition seeing the ABI
    /// easily makes code review easier.
    ///
    /// [Other ABIs]: https://doc.rust-lang.org/reference/items/external-blocks.html#abi
    pub MISSING_ABI,
    Warn,
    "No declared ABI for extern declaration"
}

declare_lint! {
    /// The `invalid_doc_attributes` lint detects when the `#[doc(...)]` is
    /// misused.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(warnings)]
    ///
    /// pub mod submodule {
    ///     #![doc(test(no_crate_inject))]
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Previously, incorrect usage of the `#[doc(..)]` attribute was not
    /// being validated. Usually these should be rejected as a hard error,
    /// but this lint was introduced to avoid breaking any existing
    /// crates which included them.
    pub INVALID_DOC_ATTRIBUTES,
    Deny,
    "detects invalid `#[doc(...)]` attributes",
}

declare_lint! {
    /// The `rust_2021_incompatible_or_patterns` lint detects usage of old versions of or-patterns.
    ///
    /// ### Example
    ///
    /// ```rust,edition2018,compile_fail
    /// #![deny(rust_2021_incompatible_or_patterns)]
    ///
    /// macro_rules! match_any {
    ///     ( $expr:expr , $( $( $pat:pat )|+ => $expr_arm:expr ),+ ) => {
    ///         match $expr {
    ///             $(
    ///                 $( $pat => $expr_arm, )+
    ///             )+
    ///         }
    ///     };
    /// }
    ///
    /// fn main() {
    ///     let result: Result<i64, i32> = Err(42);
    ///     let int: i64 = match_any!(result, Ok(i) | Err(i) => i.into());
    ///     assert_eq!(int, 42);
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// In Rust 2021, the `pat` matcher will match additional patterns, which include the `|` character.
    pub RUST_2021_INCOMPATIBLE_OR_PATTERNS,
    Allow,
    "detects usage of old versions of or-patterns",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionError(Edition::Edition2021),
        reference: "<https://doc.rust-lang.org/nightly/edition-guide/rust-2021/or-patterns-macro-rules.html>",
    };
}

declare_lint! {
    /// The `rust_2021_prelude_collisions` lint detects the usage of trait methods which are ambiguous
    /// with traits added to the prelude in future editions.
    ///
    /// ### Example
    ///
    /// ```rust,edition2018,compile_fail
    /// #![deny(rust_2021_prelude_collisions)]
    ///
    /// trait Foo {
    ///     fn try_into(self) -> Result<String, !>;
    /// }
    ///
    /// impl Foo for &str {
    ///     fn try_into(self) -> Result<String, !> {
    ///         Ok(String::from(self))
    ///     }
    /// }
    ///
    /// fn main() {
    ///     let x: String = "3".try_into().unwrap();
    ///     //                  ^^^^^^^^
    ///     // This call to try_into matches both Foo::try_into and TryInto::try_into as
    ///     // `TryInto` has been added to the Rust prelude in 2021 edition.
    ///     println!("{x}");
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// In Rust 2021, one of the important introductions is the [prelude changes], which add
    /// `TryFrom`, `TryInto`, and `FromIterator` into the standard library's prelude. Since this
    /// results in an ambiguity as to which method/function to call when an existing `try_into`
    /// method is called via dot-call syntax or a `try_from`/`from_iter` associated function
    /// is called directly on a type.
    ///
    /// [prelude changes]: https://blog.rust-lang.org/inside-rust/2021/03/04/planning-rust-2021.html#prelude-changes
    pub RUST_2021_PRELUDE_COLLISIONS,
    Allow,
    "detects the usage of trait methods which are ambiguous with traits added to the \
        prelude in future editions",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionError(Edition::Edition2021),
        reference: "<https://doc.rust-lang.org/nightly/edition-guide/rust-2021/prelude.html>",
    };
}

declare_lint! {
    /// The `rust_2024_prelude_collisions` lint detects the usage of trait methods which are ambiguous
    /// with traits added to the prelude in future editions.
    ///
    /// ### Example
    ///
    /// ```rust,edition2021,compile_fail
    /// #![deny(rust_2024_prelude_collisions)]
    /// trait Meow {
    ///     fn poll(&self) {}
    /// }
    /// impl<T> Meow for T {}
    ///
    /// fn main() {
    ///     core::pin::pin!(async {}).poll();
    ///     //                        ^^^^^^
    ///     // This call to try_into matches both Future::poll and Meow::poll as
    ///     // `Future` has been added to the Rust prelude in 2024 edition.
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Rust 2024, introduces two new additions to the standard library's prelude:
    /// `Future` and `IntoFuture`. This results in an ambiguity as to which method/function
    /// to call when an existing `poll`/`into_future` method is called via dot-call syntax or
    /// a `poll`/`into_future` associated function is called directly on a type.
    ///
    pub RUST_2024_PRELUDE_COLLISIONS,
    Allow,
    "detects the usage of trait methods which are ambiguous with traits added to the \
        prelude in future editions",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionError(Edition::Edition2024),
        reference: "<https://doc.rust-lang.org/nightly/edition-guide/rust-2024/prelude.html>",
    };
}

declare_lint! {
    /// The `rust_2021_prefixes_incompatible_syntax` lint detects identifiers that will be parsed as a
    /// prefix instead in Rust 2021.
    ///
    /// ### Example
    ///
    /// ```rust,edition2018,compile_fail
    /// #![deny(rust_2021_prefixes_incompatible_syntax)]
    ///
    /// macro_rules! m {
    ///     (z $x:expr) => ();
    /// }
    ///
    /// m!(z"hey");
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// In Rust 2015 and 2018, `z"hey"` is two tokens: the identifier `z`
    /// followed by the string literal `"hey"`. In Rust 2021, the `z` is
    /// considered a prefix for `"hey"`.
    ///
    /// This lint suggests to add whitespace between the `z` and `"hey"` tokens
    /// to keep them separated in Rust 2021.
    // Allow this lint -- rustdoc doesn't yet support threading edition into this lint's parser.
    #[allow(rustdoc::invalid_rust_codeblocks)]
    pub RUST_2021_PREFIXES_INCOMPATIBLE_SYNTAX,
    Allow,
    "identifiers that will be parsed as a prefix in Rust 2021",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionError(Edition::Edition2021),
        reference: "<https://doc.rust-lang.org/nightly/edition-guide/rust-2021/reserving-syntax.html>",
    };
    crate_level_only
}

declare_lint! {
    /// The `unsupported_fn_ptr_calling_conventions` lint is output whenever there is a use of
    /// a target dependent calling convention on a target that does not support this calling
    /// convention on a function pointer.
    ///
    /// For example `stdcall` does not make much sense for a x86_64 or, more apparently, powerpc
    /// code, because this calling convention was never specified for those targets.
    ///
    /// ### Example
    ///
    /// ```rust,ignore (needs specific targets)
    /// fn stdcall_ptr(f: extern "stdcall" fn ()) {
    ///     f()
    /// }
    /// ```
    ///
    /// This will produce:
    ///
    /// ```text
    /// warning: the calling convention `"stdcall"` is not supported on this target
    ///   --> $DIR/unsupported.rs:34:15
    ///    |
    /// LL | fn stdcall_ptr(f: extern "stdcall" fn()) {
    ///    |               ^^^^^^^^^^^^^^^^^^^^^^^^
    ///    |
    ///    = warning: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    ///    = note: for more information, see issue #130260 <https://github.com/rust-lang/rust/issues/130260>
    ///    = note: `#[warn(unsupported_fn_ptr_calling_conventions)]` on by default
    /// ```
    ///
    /// ### Explanation
    ///
    /// On most of the targets the behaviour of `stdcall` and similar calling conventions is not
    /// defined at all, but was previously accepted due to a bug in the implementation of the
    /// compiler.
    pub UNSUPPORTED_FN_PTR_CALLING_CONVENTIONS,
    Warn,
    "use of unsupported calling convention for function pointer",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::FutureReleaseErrorReportInDeps,
        reference: "issue #130260 <https://github.com/rust-lang/rust/issues/130260>",
    };
}

declare_lint! {
    /// The `break_with_label_and_loop` lint detects labeled `break` expressions with
    /// an unlabeled loop as their value expression.
    ///
    /// ### Example
    ///
    /// ```rust
    /// 'label: loop {
    ///     break 'label loop { break 42; };
    /// };
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// In Rust, loops can have a label, and `break` expressions can refer to that label to
    /// break out of specific loops (and not necessarily the innermost one). `break` expressions
    /// can also carry a value expression, which can be another loop. A labeled `break` with an
    /// unlabeled loop as its value expression is easy to confuse with an unlabeled break with
    /// a labeled loop and is thus discouraged (but allowed for compatibility); use parentheses
    /// around the loop expression to silence this warning. Unlabeled `break` expressions with
    /// labeled loops yield a hard error, which can also be silenced by wrapping the expression
    /// in parentheses.
    pub BREAK_WITH_LABEL_AND_LOOP,
    Warn,
    "`break` expression with label and unlabeled loop as value expression"
}

declare_lint! {
    /// The `non_exhaustive_omitted_patterns` lint aims to help consumers of a `#[non_exhaustive]`
    /// struct or enum who want to match all of its fields/variants explicitly.
    ///
    /// The `#[non_exhaustive]` annotation forces matches to use wildcards, so exhaustiveness
    /// checking cannot be used to ensure that all fields/variants are matched explicitly. To remedy
    /// this, this allow-by-default lint warns the user when a match mentions some but not all of
    /// the fields/variants of a `#[non_exhaustive]` struct or enum.
    ///
    /// ### Example
    ///
    /// ```rust,ignore (needs separate crate)
    /// // crate A
    /// #[non_exhaustive]
    /// pub enum Bar {
    ///     A,
    ///     B, // added variant in non breaking change
    /// }
    ///
    /// // in crate B
    /// #![feature(non_exhaustive_omitted_patterns_lint)]
    /// #[warn(non_exhaustive_omitted_patterns)]
    /// match Bar::A {
    ///     Bar::A => {},
    ///     _ => {},
    /// }
    /// ```
    ///
    /// This will produce:
    ///
    /// ```text
    /// warning: some variants are not matched explicitly
    ///    --> $DIR/reachable-patterns.rs:70:9
    ///    |
    /// LL |         match Bar::A {
    ///    |               ^ pattern `Bar::B` not covered
    ///    |
    ///  note: the lint level is defined here
    ///   --> $DIR/reachable-patterns.rs:69:16
    ///    |
    /// LL |         #[warn(non_exhaustive_omitted_patterns)]
    ///    |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ///    = help: ensure that all variants are matched explicitly by adding the suggested match arms
    ///    = note: the matched value is of type `Bar` and the `non_exhaustive_omitted_patterns` attribute was found
    /// ```
    ///
    /// Warning: setting this to `deny` will make upstream non-breaking changes (adding fields or
    /// variants to a `#[non_exhaustive]` struct or enum) break your crate. This goes against
    /// expected semver behavior.
    ///
    /// ### Explanation
    ///
    /// Structs and enums tagged with `#[non_exhaustive]` force the user to add a (potentially
    /// redundant) wildcard when pattern-matching, to allow for future addition of fields or
    /// variants. The `non_exhaustive_omitted_patterns` lint detects when such a wildcard happens to
    /// actually catch some fields/variants. In other words, when the match without the wildcard
    /// would not be exhaustive. This lets the user be informed if new fields/variants were added.
    pub NON_EXHAUSTIVE_OMITTED_PATTERNS,
    Allow,
    "detect when patterns of types marked `non_exhaustive` are missed",
    @feature_gate = non_exhaustive_omitted_patterns_lint;
}

declare_lint! {
    #[allow(text_direction_codepoint_in_literal)]
    /// The `text_direction_codepoint_in_comment` lint detects Unicode codepoints in comments that
    /// change the visual representation of text on screen in a way that does not correspond to
    /// their on memory representation.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(text_direction_codepoint_in_comment)]
    /// fn main() {
    ///     println!("{:?}"); // '‮');
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Unicode allows changing the visual flow of text on screen in order to support scripts that
    /// are written right-to-left, but a specially crafted comment can make code that will be
    /// compiled appear to be part of a comment, depending on the software used to read the code.
    /// To avoid potential problems or confusion, such as in CVE-2021-42574, by default we deny
    /// their use.
    pub TEXT_DIRECTION_CODEPOINT_IN_COMMENT,
    Deny,
    "invisible directionality-changing codepoints in comment"
}

declare_lint! {
    /// The `duplicate_macro_attributes` lint detects when a `#[test]`-like built-in macro
    /// attribute is duplicated on an item. This lint may trigger on `bench`, `cfg_eval`, `test`
    /// and `test_case`.
    ///
    /// ### Example
    ///
    /// ```rust,ignore (needs --test)
    /// #[test]
    /// #[test]
    /// fn foo() {}
    /// ```
    ///
    /// This will produce:
    ///
    /// ```text
    /// warning: duplicated attribute
    ///  --> src/lib.rs:2:1
    ///   |
    /// 2 | #[test]
    ///   | ^^^^^^^
    ///   |
    ///   = note: `#[warn(duplicate_macro_attributes)]` on by default
    /// ```
    ///
    /// ### Explanation
    ///
    /// A duplicated attribute may erroneously originate from a copy-paste and the effect of it
    /// being duplicated may not be obvious or desirable.
    ///
    /// For instance, doubling the `#[test]` attributes registers the test to be run twice with no
    /// change to its environment.
    ///
    /// [issue #90979]: https://github.com/rust-lang/rust/issues/90979
    pub DUPLICATE_MACRO_ATTRIBUTES,
    Warn,
    "duplicated attribute"
}

declare_lint! {
    /// The `deprecated_where_clause_location` lint detects when a where clause in front of the equals
    /// in an associated type.
    ///
    /// ### Example
    ///
    /// ```rust
    /// trait Trait {
    ///   type Assoc<'a> where Self: 'a;
    /// }
    ///
    /// impl Trait for () {
    ///   type Assoc<'a> where Self: 'a = ();
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The preferred location for where clauses on associated types
    /// is after the type. However, for most of generic associated types development,
    /// it was only accepted before the equals. To provide a transition period and
    /// further evaluate this change, both are currently accepted. At some point in
    /// the future, this may be disallowed at an edition boundary; but, that is
    /// undecided currently.
    pub DEPRECATED_WHERE_CLAUSE_LOCATION,
    Warn,
    "deprecated where clause location"
}

declare_lint! {
    /// The `test_unstable_lint` lint tests unstable lints and is perma-unstable.
    ///
    /// ### Example
    ///
    /// ```rust
    /// // This lint is intentionally used to test the compiler's behavior
    /// // when an unstable lint is enabled without the corresponding feature gate.
    /// #![allow(test_unstable_lint)]
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// In order to test the behavior of unstable lints, a permanently-unstable
    /// lint is required. This lint can be used to trigger warnings and errors
    /// from the compiler related to unstable lints.
    pub TEST_UNSTABLE_LINT,
    Deny,
    "this unstable lint is only for testing",
    @feature_gate = test_unstable_lint;
}

declare_lint! {
    /// The `ffi_unwind_calls` lint detects calls to foreign functions or function pointers with
    /// `C-unwind` or other FFI-unwind ABIs.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![warn(ffi_unwind_calls)]
    ///
    /// unsafe extern "C-unwind" {
    ///     fn foo();
    /// }
    ///
    /// fn bar() {
    ///     unsafe { foo(); }
    ///     let ptr: unsafe extern "C-unwind" fn() = foo;
    ///     unsafe { ptr(); }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// For crates containing such calls, if they are compiled with `-C panic=unwind` then the
    /// produced library cannot be linked with crates compiled with `-C panic=abort`. For crates
    /// that desire this ability it is therefore necessary to avoid such calls.
    pub FFI_UNWIND_CALLS,
    Allow,
    "call to foreign functions or function pointers with FFI-unwind ABI"
}

declare_lint! {
    /// The `linker_messages` lint forwards warnings from the linker.
    ///
    /// ### Example
    ///
    /// ```rust,ignore (needs CLI args, platform-specific)
    /// #[warn(linker_messages)]
    /// extern "C" {
    ///   fn foo();
    /// }
    /// fn main () { unsafe { foo(); } }
    /// ```
    ///
    /// On Linux, using `gcc -Wl,--warn-unresolved-symbols` as a linker, this will produce
    ///
    /// ```text
    /// warning: linker stderr: rust-lld: undefined symbol: foo
    ///          >>> referenced by rust_out.69edbd30df4ae57d-cgu.0
    ///          >>>               rust_out.rust_out.69edbd30df4ae57d-cgu.0.rcgu.o:(rust_out::main::h3a90094b06757803)
    ///   |
    /// note: the lint level is defined here
    ///  --> warn.rs:1:9
    ///   |
    /// 1 | #![warn(linker_messages)]
    ///   |         ^^^^^^^^^^^^^^^
    /// warning: 1 warning emitted
    /// ```
    ///
    /// ### Explanation
    ///
    /// Linkers emit platform-specific and program-specific warnings that cannot be predicted in
    /// advance by the Rust compiler. Such messages are ignored by default for now. While linker
    /// warnings could be very useful they have been ignored for many years by essentially all
    /// users, so we need to do a bit more work than just surfacing their text to produce a clear
    /// and actionable warning of similar quality to our other diagnostics. See this tracking
    /// issue for more details: <https://github.com/rust-lang/rust/issues/136096>.
    pub LINKER_MESSAGES,
    Allow,
    "warnings emitted at runtime by the target-specific linker program"
}

declare_lint! {
    /// The `named_arguments_used_positionally` lint detects cases where named arguments are only
    /// used positionally in format strings. This usage is valid but potentially very confusing.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(named_arguments_used_positionally)]
    /// fn main() {
    ///     let _x = 5;
    ///     println!("{}", _x = 1); // Prints 1, will trigger lint
    ///
    ///     println!("{}", _x); // Prints 5, no lint emitted
    ///     println!("{_x}", _x = _x); // Prints 5, no lint emitted
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Rust formatting strings can refer to named arguments by their position, but this usage is
    /// potentially confusing. In particular, readers can incorrectly assume that the declaration
    /// of named arguments is an assignment (which would produce the unit type).
    /// For backwards compatibility, this is not a hard error.
    pub NAMED_ARGUMENTS_USED_POSITIONALLY,
    Warn,
    "named arguments in format used positionally"
}

declare_lint! {
    /// The `never_type_fallback_flowing_into_unsafe` lint detects cases where never type fallback
    /// affects unsafe function calls.
    ///
    /// ### Never type fallback
    ///
    /// When the compiler sees a value of type [`!`] it implicitly inserts a coercion (if possible),
    /// to allow type check to infer any type:
    ///
    /// ```ignore (illustrative-and-has-placeholders)
    /// // this
    /// let x: u8 = panic!();
    ///
    /// // is (essentially) turned by the compiler into
    /// let x: u8 = absurd(panic!());
    ///
    /// // where absurd is a function with the following signature
    /// // (it's sound, because `!` always marks unreachable code):
    /// fn absurd<T>(never: !) -> T { ... }
    /// ```
    ///
    /// While it's convenient to be able to use non-diverging code in one of the branches (like
    /// `if a { b } else { return }`) this could lead to compilation errors:
    ///
    /// ```compile_fail
    /// // this
    /// { panic!() };
    ///
    /// // gets turned into this
    /// { absurd(panic!()) }; // error: can't infer the type of `absurd`
    /// ```
    ///
    /// To prevent such errors, compiler remembers where it inserted `absurd` calls, and if it
    /// can't infer their type, it sets the type to fallback. `{ absurd::<Fallback>(panic!()) };`.
    /// This is what is known as "never type fallback".
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(never_type_fallback_flowing_into_unsafe)]
    /// fn main() {
    ///     if true {
    ///         // return has type `!` which, is some cases, causes never type fallback
    ///         return
    ///     } else {
    ///         // `zeroed` is an unsafe function, which returns an unbounded type
    ///         unsafe { std::mem::zeroed() }
    ///     };
    ///     // depending on the fallback, `zeroed` may create `()` (which is completely sound),
    ///     // or `!` (which is instant undefined behavior)
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Due to historic reasons never type fallback was `()`, meaning that `!` got spontaneously
    /// coerced to `()`. There are plans to change that, but they may make the code such as above
    /// unsound. Instead of depending on the fallback, you should specify the type explicitly:
    /// ```
    /// if true {
    ///     return
    /// } else {
    ///     // type is explicitly specified, fallback can't hurt us no more
    ///     unsafe { std::mem::zeroed::<()>() }
    /// };
    /// ```
    ///
    /// See [Tracking Issue for making `!` fall back to `!`](https://github.com/rust-lang/rust/issues/123748).
    ///
    /// [`!`]: https://doc.rust-lang.org/core/primitive.never.html
    /// [`()`]: https://doc.rust-lang.org/core/primitive.unit.html
    pub NEVER_TYPE_FALLBACK_FLOWING_INTO_UNSAFE,
    Warn,
    "never type fallback affecting unsafe function calls",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionAndFutureReleaseSemanticsChange(Edition::Edition2024),
        reference: "<https://doc.rust-lang.org/nightly/edition-guide/rust-2024/never-type-fallback.html>",
    };
    @edition Edition2024 => Deny;
    report_in_external_macro
}

declare_lint! {
    /// The `dependency_on_unit_never_type_fallback` lint detects cases where code compiles with
    /// [never type fallback] being [`()`], but will stop compiling with fallback being [`!`].
    ///
    /// [never type fallback]: https://doc.rust-lang.org/nightly/core/primitive.never.html#never-type-fallback
    /// [`!`]: https://doc.rust-lang.org/core/primitive.never.html
    /// [`()`]: https://doc.rust-lang.org/core/primitive.unit.html
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail,edition2021
    /// #![deny(dependency_on_unit_never_type_fallback)]
    /// fn main() {
    ///     if true {
    ///         // return has type `!` which, is some cases, causes never type fallback
    ///         return
    ///     } else {
    ///         // the type produced by this call is not specified explicitly,
    ///         // so it will be inferred from the previous branch
    ///         Default::default()
    ///     };
    ///     // depending on the fallback, this may compile (because `()` implements `Default`),
    ///     // or it may not (because `!` does not implement `Default`)
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Due to historic reasons never type fallback was `()`, meaning that `!` got spontaneously
    /// coerced to `()`. There are plans to change that, but they may make the code such as above
    /// not compile. Instead of depending on the fallback, you should specify the type explicitly:
    /// ```
    /// if true {
    ///     return
    /// } else {
    ///     // type is explicitly specified, fallback can't hurt us no more
    ///     <() as Default>::default()
    /// };
    /// ```
    ///
    /// See [Tracking Issue for making `!` fall back to `!`](https://github.com/rust-lang/rust/issues/123748).
    pub DEPENDENCY_ON_UNIT_NEVER_TYPE_FALLBACK,
    Warn,
    "never type fallback affecting unsafe function calls",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionAndFutureReleaseError(Edition::Edition2024),
        reference: "<https://doc.rust-lang.org/nightly/edition-guide/rust-2024/never-type-fallback.html>",
    };
    report_in_external_macro
}

declare_lint! {
    /// The `invalid_macro_export_arguments` lint detects cases where `#[macro_export]` is being used with invalid arguments.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(invalid_macro_export_arguments)]
    ///
    /// #[macro_export(invalid_parameter)]
    /// macro_rules! myMacro {
    ///    () => {
    ///         // [...]
    ///    }
    /// }
    ///
    /// #[macro_export(too, many, items)]
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The only valid argument is `#[macro_export(local_inner_macros)]` or no argument (`#[macro_export]`).
    /// You can't have multiple arguments in a `#[macro_export(..)]`, or mention arguments other than `local_inner_macros`.
    ///
    pub INVALID_MACRO_EXPORT_ARGUMENTS,
    Warn,
    "\"invalid_parameter\" isn't a valid argument for `#[macro_export]`",
}

declare_lint! {
    /// The `private_interfaces` lint detects types in a primary interface of an item,
    /// that are more private than the item itself. Primary interface of an item is all
    /// its interface except for bounds on generic parameters and where clauses.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// # #![allow(unused)]
    /// #![deny(private_interfaces)]
    /// struct SemiPriv;
    ///
    /// mod m1 {
    ///     struct Priv;
    ///     impl crate::SemiPriv {
    ///         pub fn f(_: Priv) {}
    ///     }
    /// }
    ///
    /// # fn main() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Having something private in primary interface guarantees that
    /// the item will be unusable from outer modules due to type privacy.
    pub PRIVATE_INTERFACES,
    Warn,
    "private type in primary interface of an item",
}

declare_lint! {
    /// The `private_bounds` lint detects types in a secondary interface of an item,
    /// that are more private than the item itself. Secondary interface of an item consists of
    /// bounds on generic parameters and where clauses, including supertraits for trait items.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// # #![allow(unused)]
    /// #![deny(private_bounds)]
    ///
    /// struct PrivTy;
    /// pub struct S
    ///     where PrivTy:
    /// {}
    /// # fn main() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Having private types or traits in item bounds makes it less clear what interface
    /// the item actually provides.
    pub PRIVATE_BOUNDS,
    Warn,
    "private type in secondary interface of an item",
}

declare_lint! {
    /// The `unnameable_types` lint detects types for which you can get objects of that type,
    /// but cannot name the type itself.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// # #![allow(unused)]
    /// #![deny(unnameable_types)]
    /// mod m {
    ///     pub struct S;
    /// }
    ///
    /// pub fn get_unnameable() -> m::S { m::S }
    /// # fn main() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// It is often expected that if you can obtain an object of type `T`, then
    /// you can name the type `T` as well; this lint attempts to enforce this rule.
    /// The recommended action is to either reexport the type properly to make it nameable,
    /// or document that users are not supposed to be able to name it for one reason or another.
    ///
    /// Besides types, this lint applies to traits because traits can also leak through signatures,
    /// and you may obtain objects of their `dyn Trait` or `impl Trait` types.
    pub UNNAMEABLE_TYPES,
    Allow,
    "effective visibility of a type is larger than the area in which it can be named",
}

declare_lint! {
    /// The `unknown_or_malformed_diagnostic_attributes` lint detects unrecognized or otherwise malformed
    /// diagnostic attributes.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![feature(diagnostic_namespace)]
    /// #[diagnostic::does_not_exist]
    /// struct Foo;
    /// ```
    ///
    /// {{produces}}
    ///
    ///
    /// ### Explanation
    ///
    /// It is usually a mistake to specify a diagnostic attribute that does not exist. Check
    /// the spelling, and check the diagnostic attribute listing for the correct name. Also
    /// consider if you are using an old version of the compiler, and the attribute
    /// is only available in a newer version.
    pub UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES,
    Warn,
    "unrecognized or malformed diagnostic attribute",
}

declare_lint! {
    /// The `ambiguous_glob_imports` lint detects glob imports that should report ambiguity
    /// errors, but previously didn't do that due to rustc bugs.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(ambiguous_glob_imports)]
    /// pub fn foo() -> u32 {
    ///     use sub::*;
    ///     C
    /// }
    ///
    /// mod sub {
    ///     mod mod1 { pub const C: u32 = 1; }
    ///     mod mod2 { pub const C: u32 = 2; }
    ///
    ///     pub use mod1::*;
    ///     pub use mod2::*;
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Previous versions of Rust compile it successfully because it
    /// had lost the ambiguity error when resolve `use sub::mod2::*`.
    ///
    /// This is a [future-incompatible] lint to transition this to a
    /// hard error in the future.
    ///
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub AMBIGUOUS_GLOB_IMPORTS,
    Warn,
    "detects certain glob imports that require reporting an ambiguity error",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::FutureReleaseErrorDontReportInDeps,
        reference: "issue #114095 <https://github.com/rust-lang/rust/issues/114095>",
    };
}

declare_lint! {
    /// The `refining_impl_trait_reachable` lint detects `impl Trait` return
    /// types in method signatures that are refined by a publically reachable
    /// trait implementation, meaning the implementation adds information about
    /// the return type that is not present in the trait.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(refining_impl_trait)]
    ///
    /// use std::fmt::Display;
    ///
    /// pub trait AsDisplay {
    ///     fn as_display(&self) -> impl Display;
    /// }
    ///
    /// impl<'s> AsDisplay for &'s str {
    ///     fn as_display(&self) -> Self {
    ///         *self
    ///     }
    /// }
    ///
    /// fn main() {
    ///     // users can observe that the return type of
    ///     // `<&str as AsDisplay>::as_display()` is `&str`.
    ///     let _x: &str = "".as_display();
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Callers of methods for types where the implementation is known are
    /// able to observe the types written in the impl signature. This may be
    /// intended behavior, but may also lead to implementation details being
    /// revealed unintentionally. In particular, it may pose a semver hazard
    /// for authors of libraries who do not wish to make stronger guarantees
    /// about the types than what is written in the trait signature.
    ///
    /// `refining_impl_trait` is a lint group composed of two lints:
    ///
    /// * `refining_impl_trait_reachable`, for refinements that are publically
    ///   reachable outside a crate, and
    /// * `refining_impl_trait_internal`, for refinements that are only visible
    ///    within a crate.
    ///
    /// We are seeking feedback on each of these lints; see issue
    /// [#121718](https://github.com/rust-lang/rust/issues/121718) for more
    /// information.
    pub REFINING_IMPL_TRAIT_REACHABLE,
    Warn,
    "impl trait in impl method signature does not match trait method signature",
}

declare_lint! {
    /// The `refining_impl_trait_internal` lint detects `impl Trait` return
    /// types in method signatures that are refined by a trait implementation,
    /// meaning the implementation adds information about the return type that
    /// is not present in the trait.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(refining_impl_trait)]
    ///
    /// use std::fmt::Display;
    ///
    /// trait AsDisplay {
    ///     fn as_display(&self) -> impl Display;
    /// }
    ///
    /// impl<'s> AsDisplay for &'s str {
    ///     fn as_display(&self) -> Self {
    ///         *self
    ///     }
    /// }
    ///
    /// fn main() {
    ///     // users can observe that the return type of
    ///     // `<&str as AsDisplay>::as_display()` is `&str`.
    ///     let _x: &str = "".as_display();
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Callers of methods for types where the implementation is known are
    /// able to observe the types written in the impl signature. This may be
    /// intended behavior, but may also lead to implementation details being
    /// revealed unintentionally. In particular, it may pose a semver hazard
    /// for authors of libraries who do not wish to make stronger guarantees
    /// about the types than what is written in the trait signature.
    ///
    /// `refining_impl_trait` is a lint group composed of two lints:
    ///
    /// * `refining_impl_trait_reachable`, for refinements that are publically
    ///   reachable outside a crate, and
    /// * `refining_impl_trait_internal`, for refinements that are only visible
    ///    within a crate.
    ///
    /// We are seeking feedback on each of these lints; see issue
    /// [#121718](https://github.com/rust-lang/rust/issues/121718) for more
    /// information.
    pub REFINING_IMPL_TRAIT_INTERNAL,
    Warn,
    "impl trait in impl method signature does not match trait method signature",
}

declare_lint! {
    /// The `elided_lifetimes_in_associated_constant` lint detects elided lifetimes
    /// in associated constants when there are other lifetimes in scope. This was
    /// accidentally supported, and this lint was later relaxed to allow eliding
    /// lifetimes to `'static` when there are no lifetimes in scope.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(elided_lifetimes_in_associated_constant)]
    ///
    /// struct Foo<'a>(&'a ());
    ///
    /// impl<'a> Foo<'a> {
    ///     const STR: &str = "hello, world";
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Previous version of Rust
    ///
    /// Implicit static-in-const behavior was decided [against] for associated
    /// constants because of ambiguity. This, however, regressed and the compiler
    /// erroneously treats elided lifetimes in associated constants as lifetime
    /// parameters on the impl.
    ///
    /// This is a [future-incompatible] lint to transition this to a
    /// hard error in the future.
    ///
    /// [against]: https://github.com/rust-lang/rust/issues/38831
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub ELIDED_LIFETIMES_IN_ASSOCIATED_CONSTANT,
    Deny,
    "elided lifetimes cannot be used in associated constants in impls",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::FutureReleaseErrorDontReportInDeps,
        reference: "issue #115010 <https://github.com/rust-lang/rust/issues/115010>",
    };
}

declare_lint! {
    /// The `private_macro_use` lint detects private macros that are imported
    /// with `#[macro_use]`.
    ///
    /// ### Example
    ///
    /// ```rust,ignore (needs extern crate)
    /// // extern_macro.rs
    /// macro_rules! foo_ { () => {}; }
    /// use foo_ as foo;
    ///
    /// // code.rs
    ///
    /// #![deny(private_macro_use)]
    ///
    /// #[macro_use]
    /// extern crate extern_macro;
    ///
    /// fn main() {
    ///     foo!();
    /// }
    /// ```
    ///
    /// This will produce:
    ///
    /// ```text
    /// error: cannot find macro `foo` in this scope
    /// ```
    ///
    /// ### Explanation
    ///
    /// This lint arises from overlooking visibility checks for macros
    /// in an external crate.
    ///
    /// This is a [future-incompatible] lint to transition this to a
    /// hard error in the future.
    ///
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub PRIVATE_MACRO_USE,
    Warn,
    "detects certain macro bindings that should not be re-exported",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::FutureReleaseErrorDontReportInDeps,
        reference: "issue #120192 <https://github.com/rust-lang/rust/issues/120192>",
    };
}

declare_lint! {
    /// The `uncovered_param_in_projection` lint detects a violation of one of Rust's orphan rules for
    /// foreign trait implementations that concerns the use of type parameters inside trait associated
    /// type paths ("projections") whose output may not be a local type that is mistakenly considered
    /// to "cover" said parameters which is **unsound** and which may be rejected by a future version
    /// of the compiler.
    ///
    /// Originally reported in [#99554].
    ///
    /// [#99554]: https://github.com/rust-lang/rust/issues/99554
    ///
    /// ### Example
    ///
    /// ```rust,ignore (dependent)
    /// // dependency.rs
    /// #![crate_type = "lib"]
    ///
    /// pub trait Trait<T, U> {}
    /// ```
    ///
    /// ```edition2021,ignore (needs dependency)
    /// // dependent.rs
    /// trait Identity {
    ///     type Output;
    /// }
    ///
    /// impl<T> Identity for T {
    ///     type Output = T;
    /// }
    ///
    /// struct Local;
    ///
    /// impl<T> dependency::Trait<Local, T> for <T as Identity>::Output {}
    ///
    /// fn main() {}
    /// ```
    ///
    /// This will produce:
    ///
    /// ```text
    /// warning[E0210]: type parameter `T` must be covered by another type when it appears before the first local type (`Local`)
    ///   --> dependent.rs:11:6
    ///    |
    /// 11 | impl<T> dependency::Trait<Local, T> for <T as Identity>::Output {}
    ///    |      ^ type parameter `T` must be covered by another type when it appears before the first local type (`Local`)
    ///    |
    ///    = warning: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    ///    = note: for more information, see issue #124559 <https://github.com/rust-lang/rust/issues/124559>
    ///    = note: implementing a foreign trait is only possible if at least one of the types for which it is implemented is local, and no uncovered type parameters appear before that first local type
    ///    = note: in this case, 'before' refers to the following order: `impl<..> ForeignTrait<T1, ..., Tn> for T0`, where `T0` is the first and `Tn` is the last
    ///    = note: `#[warn(uncovered_param_in_projection)]` on by default
    /// ```
    ///
    /// ### Explanation
    ///
    /// FIXME(fmease): Write explainer.
    pub UNCOVERED_PARAM_IN_PROJECTION,
    Warn,
    "impl contains type parameters that are not covered",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::FutureReleaseErrorDontReportInDeps,
        reference: "issue #124559 <https://github.com/rust-lang/rust/issues/124559>",
    };
}

declare_lint! {
    /// The `deprecated_safe_2024` lint detects unsafe functions being used as
    /// safe functions.
    ///
    /// ### Example
    ///
    /// ```rust,edition2021,compile_fail
    /// #![deny(deprecated_safe)]
    /// // edition 2021
    /// use std::env;
    /// fn enable_backtrace() {
    ///     env::set_var("RUST_BACKTRACE", "1");
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Rust [editions] allow the language to evolve without breaking backward
    /// compatibility. This lint catches code that uses `unsafe` functions that
    /// were declared as safe (non-`unsafe`) in editions prior to Rust 2024. If
    /// you switch the compiler to Rust 2024 without updating the code, then it
    /// will fail to compile if you are using a function previously marked as
    /// safe.
    ///
    /// You can audit the code to see if it suffices the preconditions of the
    /// `unsafe` code, and if it does, you can wrap it in an `unsafe` block. If
    /// you can't fulfill the preconditions, you probably need to switch to a
    /// different way of doing what you want to achieve.
    ///
    /// This lint can automatically wrap the calls in `unsafe` blocks, but this
    /// obviously cannot verify that the preconditions of the `unsafe`
    /// functions are fulfilled, so that is still up to the user.
    ///
    /// The lint is currently "allow" by default, but that might change in the
    /// future.
    ///
    /// [editions]: https://doc.rust-lang.org/edition-guide/
    pub DEPRECATED_SAFE_2024,
    Allow,
    "detects unsafe functions being used as safe functions",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionError(Edition::Edition2024),
        reference: "<https://doc.rust-lang.org/nightly/edition-guide/rust-2024/newly-unsafe-functions.html>",
    };
}

declare_lint! {
    /// The `missing_unsafe_on_extern` lint detects missing unsafe keyword on extern declarations.
    ///
    /// ### Example
    ///
    /// ```rust,edition2021
    /// #![warn(missing_unsafe_on_extern)]
    /// #![allow(dead_code)]
    ///
    /// extern "C" {
    ///     fn foo(_: i32);
    /// }
    ///
    /// fn main() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Declaring extern items, even without ever using them, can cause Undefined Behavior. We
    /// should consider all sources of Undefined Behavior to be unsafe.
    ///
    /// This is a [future-incompatible] lint to transition this to a
    /// hard error in the future.
    ///
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub MISSING_UNSAFE_ON_EXTERN,
    Allow,
    "detects missing unsafe keyword on extern declarations",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionError(Edition::Edition2024),
        reference: "<https://doc.rust-lang.org/nightly/edition-guide/rust-2024/unsafe-extern.html>",
    };
}

declare_lint! {
    /// The `unsafe_attr_outside_unsafe` lint detects a missing unsafe keyword
    /// on attributes considered unsafe.
    ///
    /// ### Example
    ///
    /// ```rust,edition2021
    /// #![warn(unsafe_attr_outside_unsafe)]
    ///
    /// #[no_mangle]
    /// extern "C" fn foo() {}
    ///
    /// fn main() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Some attributes (e.g. `no_mangle`, `export_name`, `link_section` -- see
    /// [issue #82499] for a more complete list) are considered "unsafe" attributes.
    /// An unsafe attribute must only be used inside unsafe(...).
    ///
    /// This lint can automatically wrap the attributes in `unsafe(...)` , but this
    /// obviously cannot verify that the preconditions of the `unsafe`
    /// attributes are fulfilled, so that is still up to the user.
    ///
    /// The lint is currently "allow" by default, but that might change in the
    /// future.
    ///
    /// [editions]: https://doc.rust-lang.org/edition-guide/
    /// [issue #82499]: https://github.com/rust-lang/rust/issues/82499
    pub UNSAFE_ATTR_OUTSIDE_UNSAFE,
    Allow,
    "detects unsafe attributes outside of unsafe",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionError(Edition::Edition2024),
        reference: "<https://doc.rust-lang.org/nightly/edition-guide/rust-2024/unsafe-attributes.html>",
    };
}

declare_lint! {
    /// The `out_of_scope_macro_calls` lint detects `macro_rules` called when they are not in scope,
    /// above their definition, which may happen in key-value attributes.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![doc = in_root!()]
    ///
    /// macro_rules! in_root { () => { "" } }
    ///
    /// fn main() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The scope in which a `macro_rules` item is visible starts at that item and continues
    /// below it. This is more similar to `let` than to other items, which are in scope both above
    /// and below their definition.
    /// Due to a bug `macro_rules` were accidentally in scope inside some key-value attributes
    /// above their definition. The lint catches such cases.
    /// To address the issue turn the `macro_rules` into a regularly scoped item by importing it
    /// with `use`.
    ///
    /// This is a [future-incompatible] lint to transition this to a
    /// hard error in the future.
    ///
    /// [future-incompatible]: ../index.md#future-incompatible-lints
    pub OUT_OF_SCOPE_MACRO_CALLS,
    Warn,
    "detects out of scope calls to `macro_rules` in key-value attributes",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::FutureReleaseErrorDontReportInDeps,
        reference: "issue #124535 <https://github.com/rust-lang/rust/issues/124535>",
    };
}

declare_lint! {
    /// The `supertrait_item_shadowing_usage` lint detects when the
    /// usage of an item that is provided by both a subtrait and supertrait
    /// is shadowed, preferring the subtrait.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![feature(supertrait_item_shadowing)]
    /// #![deny(supertrait_item_shadowing_usage)]
    ///
    /// trait Upstream {
    ///     fn hello(&self) {}
    /// }
    /// impl<T> Upstream for T {}
    ///
    /// trait Downstream: Upstream {
    ///     fn hello(&self) {}
    /// }
    /// impl<T> Downstream for T {}
    ///
    /// struct MyType;
    /// MyType.hello();
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// RFC 3624 specified a heuristic in which a supertrait item would be
    /// shadowed by a subtrait item when ambiguity occurs during item
    /// selection. In order to mitigate side-effects of this happening
    /// silently, this lint detects these cases when users want to deny them
    /// or fix the call sites.
    pub SUPERTRAIT_ITEM_SHADOWING_USAGE,
    // FIXME(supertrait_item_shadowing): It is not decided if this should
    // warn by default at the call site.
    Allow,
    "detects when a supertrait item is shadowed by a subtrait item",
    @feature_gate = supertrait_item_shadowing;
}

declare_lint! {
    /// The `supertrait_item_shadowing_definition` lint detects when the
    /// definition of an item that is provided by both a subtrait and
    /// supertrait is shadowed, preferring the subtrait.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![feature(supertrait_item_shadowing)]
    /// #![deny(supertrait_item_shadowing_definition)]
    ///
    /// trait Upstream {
    ///     fn hello(&self) {}
    /// }
    /// impl<T> Upstream for T {}
    ///
    /// trait Downstream: Upstream {
    ///     fn hello(&self) {}
    /// }
    /// impl<T> Downstream for T {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// RFC 3624 specified a heuristic in which a supertrait item would be
    /// shadowed by a subtrait item when ambiguity occurs during item
    /// selection. In order to mitigate side-effects of this happening
    /// silently, this lint detects these cases when users want to deny them
    /// or fix their trait definitions.
    pub SUPERTRAIT_ITEM_SHADOWING_DEFINITION,
    // FIXME(supertrait_item_shadowing): It is not decided if this should
    // warn by default at the usage site.
    Allow,
    "detects when a supertrait item is shadowed by a subtrait item",
    @feature_gate = supertrait_item_shadowing;
}

declare_lint! {
    /// The `tail_expr_drop_order` lint looks for those values generated at the tail expression location,
    /// that runs a custom `Drop` destructor.
    /// Some of them may be dropped earlier in Edition 2024 that they used to in Edition 2021 and prior.
    /// This lint detects those cases and provides you information on those values and their custom destructor implementations.
    /// Your discretion on this information is required.
    ///
    /// ### Example
    /// ```rust,edition2021
    /// #![warn(tail_expr_drop_order)]
    /// struct Droppy(i32);
    /// impl Droppy {
    ///     fn get(&self) -> i32 {
    ///         self.0
    ///     }
    /// }
    /// impl Drop for Droppy {
    ///     fn drop(&mut self) {
    ///         // This is a custom destructor and it induces side-effects that is observable
    ///         // especially when the drop order at a tail expression changes.
    ///         println!("loud drop {}", self.0);
    ///     }
    /// }
    /// fn edition_2021() -> i32 {
    ///     let another_droppy = Droppy(0);
    ///     Droppy(1).get()
    /// }
    /// fn main() {
    ///     edition_2021();
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// In tail expression of blocks or function bodies,
    /// values of type with significant `Drop` implementation has an ill-specified drop order
    /// before Edition 2024 so that they are dropped only after dropping local variables.
    /// Edition 2024 introduces a new rule with drop orders for them,
    /// so that they are dropped first before dropping local variables.
    ///
    /// A significant `Drop::drop` destructor here refers to an explicit, arbitrary
    /// implementation of the `Drop` trait on the type, with exceptions including `Vec`,
    /// `Box`, `Rc`, `BTreeMap` and `HashMap` that are marked by the compiler otherwise
    /// so long that the generic types have no significant destructor recursively.
    /// In other words, a type has a significant drop destructor when it has a `Drop` implementation
    /// or its destructor invokes a significant destructor on a type.
    /// Since we cannot completely reason about the change by just inspecting the existence of
    /// a significant destructor, this lint remains only a suggestion and is set to `allow` by default.
    ///
    /// This lint only points out the issue with `Droppy`, which will be dropped before `another_droppy`
    /// does in Edition 2024.
    /// No fix will be proposed by this lint.
    /// However, the most probable fix is to hoist `Droppy` into its own local variable binding.
    /// ```rust
    /// struct Droppy(i32);
    /// impl Droppy {
    ///     fn get(&self) -> i32 {
    ///         self.0
    ///     }
    /// }
    /// fn edition_2024() -> i32 {
    ///     let value = Droppy(0);
    ///     let another_droppy = Droppy(1);
    ///     value.get()
    /// }
    /// ```
    pub TAIL_EXPR_DROP_ORDER,
    Allow,
    "Detect and warn on significant change in drop order in tail expression location",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionSemanticsChange(Edition::Edition2024),
        reference: "<https://doc.rust-lang.org/nightly/edition-guide/rust-2024/temporary-tail-expr-scope.html>",
    };
}

declare_lint! {
    /// The `rust_2024_guarded_string_incompatible_syntax` lint detects `#` tokens
    /// that will be parsed as part of a guarded string literal in Rust 2024.
    ///
    /// ### Example
    ///
    /// ```rust,edition2021,compile_fail
    /// #![deny(rust_2024_guarded_string_incompatible_syntax)]
    ///
    /// macro_rules! m {
    ///     (# $x:expr #) => ();
    ///     (# $x:expr) => ();
    /// }
    ///
    /// m!(#"hey"#);
    /// m!(#"hello");
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Prior to Rust 2024, `#"hey"#` is three tokens: the first `#`
    /// followed by the string literal `"hey"` then the final `#`.
    /// In Rust 2024, the whole sequence is considered a single token.
    ///
    /// This lint suggests to add whitespace between the leading `#`
    /// and the string to keep them separated in Rust 2024.
    // Allow this lint -- rustdoc doesn't yet support threading edition into this lint's parser.
    #[allow(rustdoc::invalid_rust_codeblocks)]
    pub RUST_2024_GUARDED_STRING_INCOMPATIBLE_SYNTAX,
    Allow,
    "will be parsed as a guarded string in Rust 2024",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionError(Edition::Edition2024),
        reference: "<https://doc.rust-lang.org/nightly/edition-guide/rust-2024/reserved-syntax.html>",
    };
    crate_level_only
}

declare_lint! {
    /// The `wasm_c_abi` lint detects usage of the `extern "C"` ABI of wasm that is affected
    /// by a planned ABI change that has the goal of aligning Rust with the standard C ABI
    /// of this target.
    ///
    /// ### Example
    ///
    /// ```rust,ignore (needs wasm32-unknown-unknown)
    /// #[repr(C)]
    /// struct MyType(i32, i32);
    ///
    /// extern "C" my_fun(x: MyType) {}
    /// ```
    ///
    /// This will produce:
    ///
    /// ```text
    /// error: this function function definition is affected by the wasm ABI transition: it passes an argument of non-scalar type `MyType`
    ///   --> $DIR/wasm_c_abi_transition.rs:17:1
    ///    |
    ///    | pub extern "C" fn my_fun(_x: MyType) {}
    ///    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ///    |
    ///    = warning: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    ///    = note: for more information, see issue #138762 <https://github.com/rust-lang/rust/issues/138762>
    ///    = help: the "C" ABI Rust uses on wasm32-unknown-unknown will change to align with the standard "C" ABI for this target
    /// ```
    ///
    /// ### Explanation
    ///
    /// Rust has historically implemented a non-spec-compliant C ABI on wasm32-unknown-unknown. This
    /// has caused incompatibilities with other compilers and Wasm targets. In a future version
    /// of Rust, this will be fixed, and therefore code relying on the non-spec-compliant C ABI will
    /// stop functioning.
    pub WASM_C_ABI,
    Warn,
    "detects code relying on rustc's non-spec-compliant wasm C ABI",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::FutureReleaseErrorReportInDeps,
        reference: "issue #138762 <https://github.com/rust-lang/rust/issues/138762>",
    };
}

declare_lint! {
    /// The `aarch64_softfloat_neon` lint detects usage of `#[target_feature(enable = "neon")]` on
    /// softfloat aarch64 targets. Enabling this target feature causes LLVM to alter the ABI of
    /// function calls, making this attribute unsound to use.
    ///
    /// ### Example
    ///
    /// ```rust,ignore (needs aarch64-unknown-none-softfloat)
    /// #[target_feature(enable = "neon")]
    /// fn with_neon() {}
    /// ```
    ///
    /// This will produce:
    ///
    /// ```text
    /// error: enabling the `neon` target feature on the current target is unsound due to ABI issues
    ///   --> $DIR/abi-incompatible-target-feature-attribute-fcw.rs:11:18
    ///    |
    ///    | #[target_feature(enable = "neon")]
    ///    |                  ^^^^^^^^^^^^^^^
    ///    |
    ///    = warning: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    ///    = note: for more information, see issue #134375 <https://github.com/rust-lang/rust/issues/134375>
    /// ```
    ///
    /// ### Explanation
    ///
    /// If a function like `with_neon` above ends up containing calls to LLVM builtins, those will
    /// not use the correct ABI. This is caused by a lack of support in LLVM for mixing code with
    /// and without the `neon` target feature. The target feature should never have been stabilized
    /// on this target due to this issue, but the problem was not known at the time of
    /// stabilization.
    pub AARCH64_SOFTFLOAT_NEON,
    Warn,
    "detects code that could be affected by ABI issues on aarch64 softfloat targets",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::FutureReleaseErrorReportInDeps,
        reference: "issue #134375 <https://github.com/rust-lang/rust/issues/134375>",
    };
}
