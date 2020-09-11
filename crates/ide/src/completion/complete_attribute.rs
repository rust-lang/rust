//! Completion for attributes
//!
//! This module uses a bit of static metadata to provide completions
//! for built-in attributes.

use rustc_hash::FxHashSet;
use syntax::{ast, AstNode, SyntaxKind};

use crate::completion::{
    completion_context::CompletionContext,
    completion_item::{CompletionItem, CompletionItemKind, CompletionKind, Completions},
    generated_features::FEATURES,
};

pub(super) fn complete_attribute(acc: &mut Completions, ctx: &CompletionContext) -> Option<()> {
    if ctx.mod_declaration_under_caret.is_some() {
        return None;
    }

    let attribute = ctx.attribute_under_caret.as_ref()?;
    match (attribute.path(), attribute.token_tree()) {
        (Some(path), Some(token_tree)) if path.to_string() == "derive" => {
            complete_derive(acc, ctx, token_tree)
        }
        (Some(path), Some(token_tree)) if path.to_string() == "feature" => {
            complete_lint(acc, ctx, token_tree, FEATURES)
        }
        (Some(path), Some(token_tree))
            if ["allow", "warn", "deny", "forbid"]
                .iter()
                .any(|lint_level| lint_level == &path.to_string()) =>
        {
            complete_lint(acc, ctx, token_tree, DEFAULT_LINT_COMPLETIONS)
        }
        (_, Some(_token_tree)) => {}
        _ => complete_attribute_start(acc, ctx, attribute),
    }
    Some(())
}

fn complete_attribute_start(acc: &mut Completions, ctx: &CompletionContext, attribute: &ast::Attr) {
    for attr_completion in ATTRIBUTES {
        let mut item = CompletionItem::new(
            CompletionKind::Attribute,
            ctx.source_range(),
            attr_completion.label,
        )
        .kind(CompletionItemKind::Attribute);

        if let Some(lookup) = attr_completion.lookup {
            item = item.lookup_by(lookup);
        }

        match (attr_completion.snippet, ctx.config.snippet_cap) {
            (Some(snippet), Some(cap)) => {
                item = item.insert_snippet(cap, snippet);
            }
            _ => {}
        }

        if attribute.kind() == ast::AttrKind::Inner || !attr_completion.prefer_inner {
            acc.add(item);
        }
    }
}

struct AttrCompletion {
    label: &'static str,
    lookup: Option<&'static str>,
    snippet: Option<&'static str>,
    prefer_inner: bool,
}

impl AttrCompletion {
    const fn prefer_inner(self) -> AttrCompletion {
        AttrCompletion { prefer_inner: true, ..self }
    }
}

const fn attr(
    label: &'static str,
    lookup: Option<&'static str>,
    snippet: Option<&'static str>,
) -> AttrCompletion {
    AttrCompletion { label, lookup, snippet, prefer_inner: false }
}

const ATTRIBUTES: &[AttrCompletion] = &[
    attr("allow(…)", Some("allow"), Some("allow(${0:lint})")),
    attr("cfg_attr(…)", Some("cfg_attr"), Some("cfg_attr(${1:predicate}, ${0:attr})")),
    attr("cfg(…)", Some("cfg"), Some("cfg(${0:predicate})")),
    attr("deny(…)", Some("deny"), Some("deny(${0:lint})")),
    attr(r#"deprecated = "…""#, Some("deprecated"), Some(r#"deprecated = "${0:reason}""#)),
    attr("derive(…)", Some("derive"), Some(r#"derive(${0:Debug})"#)),
    attr(r#"doc = "…""#, Some("doc"), Some(r#"doc = "${0:docs}""#)),
    attr("feature(…)", Some("feature"), Some("feature(${0:flag})")).prefer_inner(),
    attr("forbid(…)", Some("forbid"), Some("forbid(${0:lint})")),
    // FIXME: resolve through macro resolution?
    attr("global_allocator", None, None).prefer_inner(),
    attr(r#"ignore = "…""#, Some("ignore"), Some(r#"ignore = "${0:reason}""#)),
    attr("inline(…)", Some("inline"), Some("inline(${0:lint})")),
    attr(r#"link_name = "…""#, Some("link_name"), Some(r#"link_name = "${0:symbol_name}""#)),
    attr("link", None, None),
    attr("macro_export", None, None),
    attr("macro_use", None, None),
    attr(r#"must_use = "…""#, Some("must_use"), Some(r#"must_use = "${0:reason}""#)),
    attr("no_mangle", None, None),
    attr("no_std", None, None).prefer_inner(),
    attr("non_exhaustive", None, None),
    attr("panic_handler", None, None).prefer_inner(),
    attr("path = \"…\"", Some("path"), Some("path =\"${0:path}\"")),
    attr("proc_macro", None, None),
    attr("proc_macro_attribute", None, None),
    attr("proc_macro_derive(…)", Some("proc_macro_derive"), Some("proc_macro_derive(${0:Trait})")),
    attr("recursion_limit = …", Some("recursion_limit"), Some("recursion_limit = ${0:128}"))
        .prefer_inner(),
    attr("repr(…)", Some("repr"), Some("repr(${0:C})")),
    attr(
        "should_panic(…)",
        Some("should_panic"),
        Some(r#"should_panic(expected = "${0:reason}")"#),
    ),
    attr(
        r#"target_feature = "…""#,
        Some("target_feature"),
        Some("target_feature = \"${0:feature}\""),
    ),
    attr("test", None, None),
    attr("used", None, None),
    attr("warn(…)", Some("warn"), Some("warn(${0:lint})")),
    attr(
        r#"windows_subsystem = "…""#,
        Some("windows_subsystem"),
        Some(r#"windows_subsystem = "${0:subsystem}""#),
    )
    .prefer_inner(),
];

fn complete_derive(acc: &mut Completions, ctx: &CompletionContext, derive_input: ast::TokenTree) {
    if let Ok(existing_derives) = parse_comma_sep_input(derive_input) {
        for derive_completion in DEFAULT_DERIVE_COMPLETIONS
            .into_iter()
            .filter(|completion| !existing_derives.contains(completion.label))
        {
            let mut label = derive_completion.label.to_owned();
            for dependency in derive_completion
                .dependencies
                .into_iter()
                .filter(|&&dependency| !existing_derives.contains(dependency))
            {
                label.push_str(", ");
                label.push_str(dependency);
            }
            acc.add(
                CompletionItem::new(CompletionKind::Attribute, ctx.source_range(), label)
                    .kind(CompletionItemKind::Attribute),
            );
        }

        for custom_derive_name in get_derive_names_in_scope(ctx).difference(&existing_derives) {
            acc.add(
                CompletionItem::new(
                    CompletionKind::Attribute,
                    ctx.source_range(),
                    custom_derive_name,
                )
                .kind(CompletionItemKind::Attribute),
            );
        }
    }
}

fn complete_lint(
    acc: &mut Completions,
    ctx: &CompletionContext,
    derive_input: ast::TokenTree,
    lints_completions: &[LintCompletion],
) {
    if let Ok(existing_lints) = parse_comma_sep_input(derive_input) {
        for lint_completion in lints_completions
            .into_iter()
            .filter(|completion| !existing_lints.contains(completion.label))
        {
            acc.add(
                CompletionItem::new(
                    CompletionKind::Attribute,
                    ctx.source_range(),
                    lint_completion.label,
                )
                .kind(CompletionItemKind::Attribute)
                .detail(lint_completion.description),
            );
        }
    }
}

fn parse_comma_sep_input(derive_input: ast::TokenTree) -> Result<FxHashSet<String>, ()> {
    match (derive_input.left_delimiter_token(), derive_input.right_delimiter_token()) {
        (Some(left_paren), Some(right_paren))
            if left_paren.kind() == SyntaxKind::L_PAREN
                && right_paren.kind() == SyntaxKind::R_PAREN =>
        {
            let mut input_derives = FxHashSet::default();
            let mut current_derive = String::new();
            for token in derive_input
                .syntax()
                .children_with_tokens()
                .filter_map(|token| token.into_token())
                .skip_while(|token| token != &left_paren)
                .skip(1)
                .take_while(|token| token != &right_paren)
            {
                if SyntaxKind::COMMA == token.kind() {
                    if !current_derive.is_empty() {
                        input_derives.insert(current_derive);
                        current_derive = String::new();
                    }
                } else {
                    current_derive.push_str(token.to_string().trim());
                }
            }

            if !current_derive.is_empty() {
                input_derives.insert(current_derive);
            }
            Ok(input_derives)
        }
        _ => Err(()),
    }
}

fn get_derive_names_in_scope(ctx: &CompletionContext) -> FxHashSet<String> {
    let mut result = FxHashSet::default();
    ctx.scope.process_all_names(&mut |name, scope_def| {
        if let hir::ScopeDef::MacroDef(mac) = scope_def {
            if mac.is_derive_macro() {
                result.insert(name.to_string());
            }
        }
    });
    result
}

struct DeriveCompletion {
    label: &'static str,
    dependencies: &'static [&'static str],
}

/// Standard Rust derives and the information about their dependencies
/// (the dependencies are needed so that the main derive don't break the compilation when added)
#[rustfmt::skip]
const DEFAULT_DERIVE_COMPLETIONS: &[DeriveCompletion] = &[
    DeriveCompletion { label: "Clone", dependencies: &[] },
    DeriveCompletion { label: "Copy", dependencies: &["Clone"] },
    DeriveCompletion { label: "Debug", dependencies: &[] },
    DeriveCompletion { label: "Default", dependencies: &[] },
    DeriveCompletion { label: "Hash", dependencies: &[] },
    DeriveCompletion { label: "PartialEq", dependencies: &[] },
    DeriveCompletion { label: "Eq", dependencies: &["PartialEq"] },
    DeriveCompletion { label: "PartialOrd", dependencies: &["PartialEq"] },
    DeriveCompletion { label: "Ord", dependencies: &["PartialOrd", "Eq", "PartialEq"] },
];

pub(super) struct LintCompletion {
    pub(super) label: &'static str,
    pub(super) description: &'static str,
}

#[rustfmt::skip]
const DEFAULT_LINT_COMPLETIONS: &[LintCompletion] = &[
    LintCompletion { label: "absolute_paths_not_starting_with_crate", description: r#"fully qualified paths that start with a module name instead of `crate`, `self`, or an extern crate name"# },
    LintCompletion { label: "anonymous_parameters", description: r#"detects anonymous parameters"# },
    LintCompletion { label: "box_pointers", description: r#"use of owned (Box type) heap memory"# },
    LintCompletion { label: "deprecated_in_future", description: r#"detects use of items that will be deprecated in a future version"# },
    LintCompletion { label: "elided_lifetimes_in_paths", description: r#"hidden lifetime parameters in types are deprecated"# },
    LintCompletion { label: "explicit_outlives_requirements", description: r#"outlives requirements can be inferred"# },
    LintCompletion { label: "indirect_structural_match", description: r#"pattern with const indirectly referencing non-structural-match type"# },
    LintCompletion { label: "keyword_idents", description: r#"detects edition keywords being used as an identifier"# },
    LintCompletion { label: "macro_use_extern_crate", description: r#"the `#[macro_use]` attribute is now deprecated in favor of using macros via the module system"# },
    LintCompletion { label: "meta_variable_misuse", description: r#"possible meta-variable misuse at macro definition"# },
    LintCompletion { label: "missing_copy_implementations", description: r#"detects potentially-forgotten implementations of `Copy`"# },
    LintCompletion { label: "missing_crate_level_docs", description: r#"detects crates with no crate-level documentation"# },
    LintCompletion { label: "missing_debug_implementations", description: r#"detects missing implementations of Debug"# },
    LintCompletion { label: "missing_docs", description: r#"detects missing documentation for public members"# },
    LintCompletion { label: "missing_doc_code_examples", description: r#"detects publicly-exported items without code samples in their documentation"# },
    LintCompletion { label: "non_ascii_idents", description: r#"detects non-ASCII identifiers"# },
    LintCompletion { label: "private_doc_tests", description: r#"detects code samples in docs of private items not documented by rustdoc"# },
    LintCompletion { label: "single_use_lifetimes", description: r#"detects lifetime parameters that are only used once"# },
    LintCompletion { label: "trivial_casts", description: r#"detects trivial casts which could be removed"# },
    LintCompletion { label: "trivial_numeric_casts", description: r#"detects trivial casts of numeric types which could be removed"# },
    LintCompletion { label: "unaligned_references", description: r#"detects unaligned references to fields of packed structs"# },
    LintCompletion { label: "unreachable_pub", description: r#"`pub` items not reachable from crate root"# },
    LintCompletion { label: "unsafe_code", description: r#"usage of `unsafe` code"# },
    LintCompletion { label: "unsafe_op_in_unsafe_fn", description: r#"unsafe operations in unsafe functions without an explicit unsafe block are deprecated"# },
    LintCompletion { label: "unstable_features", description: r#"enabling unstable features (deprecated. do not use)"# },
    LintCompletion { label: "unused_crate_dependencies", description: r#"crate dependencies that are never used"# },
    LintCompletion { label: "unused_extern_crates", description: r#"extern crates that are never used"# },
    LintCompletion { label: "unused_import_braces", description: r#"unnecessary braces around an imported item"# },
    LintCompletion { label: "unused_lifetimes", description: r#"detects lifetime parameters that are never used"# },
    LintCompletion { label: "unused_qualifications", description: r#"detects unnecessarily qualified names"# },
    LintCompletion { label: "unused_results", description: r#"unused result of an expression in a statement"# },
    LintCompletion { label: "variant_size_differences", description: r#"detects enums with widely varying variant sizes"# },
    LintCompletion { label: "array_into_iter", description: r#"detects calling `into_iter` on arrays"# },
    LintCompletion { label: "asm_sub_register", description: r#"using only a subset of a register for inline asm inputs"# },
    LintCompletion { label: "bare_trait_objects", description: r#"suggest using `dyn Trait` for trait objects"# },
    LintCompletion { label: "bindings_with_variant_name", description: r#"detects pattern bindings with the same name as one of the matched variants"# },
    LintCompletion { label: "cenum_impl_drop_cast", description: r#"a C-like enum implementing Drop is cast"# },
    LintCompletion { label: "clashing_extern_declarations", description: r#"detects when an extern fn has been declared with the same name but different types"# },
    LintCompletion { label: "coherence_leak_check", description: r#"distinct impls distinguished only by the leak-check code"# },
    LintCompletion { label: "confusable_idents", description: r#"detects visually confusable pairs between identifiers"# },
    LintCompletion { label: "dead_code", description: r#"detect unused, unexported items"# },
    LintCompletion { label: "deprecated", description: r#"detects use of deprecated items"# },
    LintCompletion { label: "ellipsis_inclusive_range_patterns", description: r#"`...` range patterns are deprecated"# },
    LintCompletion { label: "exported_private_dependencies", description: r#"public interface leaks type from a private dependency"# },
    LintCompletion { label: "illegal_floating_point_literal_pattern", description: r#"floating-point literals cannot be used in patterns"# },
    LintCompletion { label: "improper_ctypes", description: r#"proper use of libc types in foreign modules"# },
    LintCompletion { label: "improper_ctypes_definitions", description: r#"proper use of libc types in foreign item definitions"# },
    LintCompletion { label: "incomplete_features", description: r#"incomplete features that may function improperly in some or all cases"# },
    LintCompletion { label: "inline_no_sanitize", description: r#"detects incompatible use of `#[inline(always)]` and `#[no_sanitize(...)]`"# },
    LintCompletion { label: "intra_doc_link_resolution_failure", description: r#"failures in resolving intra-doc link targets"# },
    LintCompletion { label: "invalid_codeblock_attributes", description: r#"codeblock attribute looks a lot like a known one"# },
    LintCompletion { label: "invalid_value", description: r#"an invalid value is being created (such as a NULL reference)"# },
    LintCompletion { label: "irrefutable_let_patterns", description: r#"detects irrefutable patterns in if-let and while-let statements"# },
    LintCompletion { label: "late_bound_lifetime_arguments", description: r#"detects generic lifetime arguments in path segments with late bound lifetime parameters"# },
    LintCompletion { label: "mixed_script_confusables", description: r#"detects Unicode scripts whose mixed script confusables codepoints are solely used"# },
    LintCompletion { label: "mutable_borrow_reservation_conflict", description: r#"reservation of a two-phased borrow conflicts with other shared borrows"# },
    LintCompletion { label: "non_camel_case_types", description: r#"types, variants, traits and type parameters should have camel case names"# },
    LintCompletion { label: "non_shorthand_field_patterns", description: r#"using `Struct { x: x }` instead of `Struct { x }` in a pattern"# },
    LintCompletion { label: "non_snake_case", description: r#"variables, methods, functions, lifetime parameters and modules should have snake case names"# },
    LintCompletion { label: "non_upper_case_globals", description: r#"static constants should have uppercase identifiers"# },
    LintCompletion { label: "no_mangle_generic_items", description: r#"generic items must be mangled"# },
    LintCompletion { label: "overlapping_patterns", description: r#"detects overlapping patterns"# },
    LintCompletion { label: "path_statements", description: r#"path statements with no effect"# },
    LintCompletion { label: "private_in_public", description: r#"detect private items in public interfaces not caught by the old implementation"# },
    LintCompletion { label: "proc_macro_derive_resolution_fallback", description: r#"detects proc macro derives using inaccessible names from parent modules"# },
    LintCompletion { label: "redundant_semicolons", description: r#"detects unnecessary trailing semicolons"# },
    LintCompletion { label: "renamed_and_removed_lints", description: r#"lints that have been renamed or removed"# },
    LintCompletion { label: "safe_packed_borrows", description: r#"safe borrows of fields of packed structs were erroneously allowed"# },
    LintCompletion { label: "stable_features", description: r#"stable features found in `#[feature]` directive"# },
    LintCompletion { label: "trivial_bounds", description: r#"these bounds don't depend on an type parameters"# },
    LintCompletion { label: "type_alias_bounds", description: r#"bounds in type aliases are not enforced"# },
    LintCompletion { label: "tyvar_behind_raw_pointer", description: r#"raw pointer to an inference variable"# },
    LintCompletion { label: "uncommon_codepoints", description: r#"detects uncommon Unicode codepoints in identifiers"# },
    LintCompletion { label: "unconditional_recursion", description: r#"functions that cannot return without calling themselves"# },
    LintCompletion { label: "unknown_lints", description: r#"unrecognized lint attribute"# },
    LintCompletion { label: "unnameable_test_items", description: r#"detects an item that cannot be named being marked as `#[test_case]`"# },
    LintCompletion { label: "unreachable_code", description: r#"detects unreachable code paths"# },
    LintCompletion { label: "unreachable_patterns", description: r#"detects unreachable patterns"# },
    LintCompletion { label: "unstable_name_collisions", description: r#"detects name collision with an existing but unstable method"# },
    LintCompletion { label: "unused_allocation", description: r#"detects unnecessary allocations that can be eliminated"# },
    LintCompletion { label: "unused_assignments", description: r#"detect assignments that will never be read"# },
    LintCompletion { label: "unused_attributes", description: r#"detects attributes that were not used by the compiler"# },
    LintCompletion { label: "unused_braces", description: r#"unnecessary braces around an expression"# },
    LintCompletion { label: "unused_comparisons", description: r#"comparisons made useless by limits of the types involved"# },
    LintCompletion { label: "unused_doc_comments", description: r#"detects doc comments that aren't used by rustdoc"# },
    LintCompletion { label: "unused_features", description: r#"unused features found in crate-level `#[feature]` directives"# },
    LintCompletion { label: "unused_imports", description: r#"imports that are never used"# },
    LintCompletion { label: "unused_labels", description: r#"detects labels that are never used"# },
    LintCompletion { label: "unused_macros", description: r#"detects macros that were not used"# },
    LintCompletion { label: "unused_must_use", description: r#"unused result of a type flagged as `#[must_use]`"# },
    LintCompletion { label: "unused_mut", description: r#"detect mut variables which don't need to be mutable"# },
    LintCompletion { label: "unused_parens", description: r#"`if`, `match`, `while` and `return` do not need parentheses"# },
    LintCompletion { label: "unused_unsafe", description: r#"unnecessary use of an `unsafe` block"# },
    LintCompletion { label: "unused_variables", description: r#"detect variables which are not used in any way"# },
    LintCompletion { label: "warnings", description: r#"mass-change the level for lints which produce warnings"# },
    LintCompletion { label: "where_clauses_object_safety", description: r#"checks the object safety of where clauses"# },
    LintCompletion { label: "while_true", description: r#"suggest using `loop { }` instead of `while true { }`"# },
    LintCompletion { label: "ambiguous_associated_items", description: r#"ambiguous associated items"# },
    LintCompletion { label: "arithmetic_overflow", description: r#"arithmetic operation overflows"# },
    LintCompletion { label: "conflicting_repr_hints", description: r#"conflicts between `#[repr(..)]` hints that were previously accepted and used in practice"# },
    LintCompletion { label: "const_err", description: r#"constant evaluation detected erroneous expression"# },
    LintCompletion { label: "ill_formed_attribute_input", description: r#"ill-formed attribute inputs that were previously accepted and used in practice"# },
    LintCompletion { label: "incomplete_include", description: r#"trailing content in included file"# },
    LintCompletion { label: "invalid_type_param_default", description: r#"type parameter default erroneously allowed in invalid location"# },
    LintCompletion { label: "macro_expanded_macro_exports_accessed_by_absolute_paths", description: r#"macro-expanded `macro_export` macros from the current crate cannot be referred to by absolute paths"# },
    LintCompletion { label: "missing_fragment_specifier", description: r#"detects missing fragment specifiers in unused `macro_rules!` patterns"# },
    LintCompletion { label: "mutable_transmutes", description: r#"mutating transmuted &mut T from &T may cause undefined behavior"# },
    LintCompletion { label: "no_mangle_const_items", description: r#"const items will not have their symbols exported"# },
    LintCompletion { label: "order_dependent_trait_objects", description: r#"trait-object types were treated as different depending on marker-trait order"# },
    LintCompletion { label: "overflowing_literals", description: r#"literal out of range for its type"# },
    LintCompletion { label: "patterns_in_fns_without_body", description: r#"patterns in functions without body were erroneously allowed"# },
    LintCompletion { label: "pub_use_of_private_extern_crate", description: r#"detect public re-exports of private extern crates"# },
    LintCompletion { label: "soft_unstable", description: r#"a feature gate that doesn't break dependent crates"# },
    LintCompletion { label: "unconditional_panic", description: r#"operation will cause a panic at runtime"# },
    LintCompletion { label: "unknown_crate_types", description: r#"unknown crate type found in `#[crate_type]` directive"# },
];

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::completion::{test_utils::completion_list, CompletionKind};

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture, CompletionKind::Attribute);
        expect.assert_eq(&actual);
    }

    #[test]
    fn empty_derive_completion() {
        check(
            r#"
#[derive(<|>)]
struct Test {}
        "#,
            expect![[r#"
                at Clone
                at Copy, Clone
                at Debug
                at Default
                at Eq, PartialEq
                at Hash
                at Ord, PartialOrd, Eq, PartialEq
                at PartialEq
                at PartialOrd, PartialEq
            "#]],
        );
    }

    #[test]
    fn empty_lint_completion() {
        check(
            r#"#[allow(<|>)]"#,
            expect![[r#"
                at absolute_paths_not_starting_with_crate fully qualified paths that start with a module name instead of `crate`, `self`, or an extern crate name
                at ambiguous_associated_items ambiguous associated items
                at anonymous_parameters detects anonymous parameters
                at arithmetic_overflow arithmetic operation overflows
                at array_into_iter  detects calling `into_iter` on arrays
                at asm_sub_register using only a subset of a register for inline asm inputs
                at bare_trait_objects suggest using `dyn Trait` for trait objects
                at bindings_with_variant_name detects pattern bindings with the same name as one of the matched variants
                at box_pointers     use of owned (Box type) heap memory
                at cenum_impl_drop_cast a C-like enum implementing Drop is cast
                at clashing_extern_declarations detects when an extern fn has been declared with the same name but different types
                at coherence_leak_check distinct impls distinguished only by the leak-check code
                at conflicting_repr_hints conflicts between `#[repr(..)]` hints that were previously accepted and used in practice
                at confusable_idents detects visually confusable pairs between identifiers
                at const_err        constant evaluation detected erroneous expression
                at dead_code        detect unused, unexported items
                at deprecated       detects use of deprecated items
                at deprecated_in_future detects use of items that will be deprecated in a future version
                at elided_lifetimes_in_paths hidden lifetime parameters in types are deprecated
                at ellipsis_inclusive_range_patterns `...` range patterns are deprecated
                at explicit_outlives_requirements outlives requirements can be inferred
                at exported_private_dependencies public interface leaks type from a private dependency
                at ill_formed_attribute_input ill-formed attribute inputs that were previously accepted and used in practice
                at illegal_floating_point_literal_pattern floating-point literals cannot be used in patterns
                at improper_ctypes  proper use of libc types in foreign modules
                at improper_ctypes_definitions proper use of libc types in foreign item definitions
                at incomplete_features incomplete features that may function improperly in some or all cases
                at incomplete_include trailing content in included file
                at indirect_structural_match pattern with const indirectly referencing non-structural-match type
                at inline_no_sanitize detects incompatible use of `#[inline(always)]` and `#[no_sanitize(...)]`
                at intra_doc_link_resolution_failure failures in resolving intra-doc link targets
                at invalid_codeblock_attributes codeblock attribute looks a lot like a known one
                at invalid_type_param_default type parameter default erroneously allowed in invalid location
                at invalid_value    an invalid value is being created (such as a NULL reference)
                at irrefutable_let_patterns detects irrefutable patterns in if-let and while-let statements
                at keyword_idents   detects edition keywords being used as an identifier
                at late_bound_lifetime_arguments detects generic lifetime arguments in path segments with late bound lifetime parameters
                at macro_expanded_macro_exports_accessed_by_absolute_paths macro-expanded `macro_export` macros from the current crate cannot be referred to by absolute paths
                at macro_use_extern_crate the `#[macro_use]` attribute is now deprecated in favor of using macros via the module system
                at meta_variable_misuse possible meta-variable misuse at macro definition
                at missing_copy_implementations detects potentially-forgotten implementations of `Copy`
                at missing_crate_level_docs detects crates with no crate-level documentation
                at missing_debug_implementations detects missing implementations of Debug
                at missing_doc_code_examples detects publicly-exported items without code samples in their documentation
                at missing_docs     detects missing documentation for public members
                at missing_fragment_specifier detects missing fragment specifiers in unused `macro_rules!` patterns
                at mixed_script_confusables detects Unicode scripts whose mixed script confusables codepoints are solely used
                at mutable_borrow_reservation_conflict reservation of a two-phased borrow conflicts with other shared borrows
                at mutable_transmutes mutating transmuted &mut T from &T may cause undefined behavior
                at no_mangle_const_items const items will not have their symbols exported
                at no_mangle_generic_items generic items must be mangled
                at non_ascii_idents detects non-ASCII identifiers
                at non_camel_case_types types, variants, traits and type parameters should have camel case names
                at non_shorthand_field_patterns using `Struct { x: x }` instead of `Struct { x }` in a pattern
                at non_snake_case   variables, methods, functions, lifetime parameters and modules should have snake case names
                at non_upper_case_globals static constants should have uppercase identifiers
                at order_dependent_trait_objects trait-object types were treated as different depending on marker-trait order
                at overflowing_literals literal out of range for its type
                at overlapping_patterns detects overlapping patterns
                at path_statements  path statements with no effect
                at patterns_in_fns_without_body patterns in functions without body were erroneously allowed
                at private_doc_tests detects code samples in docs of private items not documented by rustdoc
                at private_in_public detect private items in public interfaces not caught by the old implementation
                at proc_macro_derive_resolution_fallback detects proc macro derives using inaccessible names from parent modules
                at pub_use_of_private_extern_crate detect public re-exports of private extern crates
                at redundant_semicolons detects unnecessary trailing semicolons
                at renamed_and_removed_lints lints that have been renamed or removed
                at safe_packed_borrows safe borrows of fields of packed structs were erroneously allowed
                at single_use_lifetimes detects lifetime parameters that are only used once
                at soft_unstable    a feature gate that doesn't break dependent crates
                at stable_features  stable features found in `#[feature]` directive
                at trivial_bounds   these bounds don't depend on an type parameters
                at trivial_casts    detects trivial casts which could be removed
                at trivial_numeric_casts detects trivial casts of numeric types which could be removed
                at type_alias_bounds bounds in type aliases are not enforced
                at tyvar_behind_raw_pointer raw pointer to an inference variable
                at unaligned_references detects unaligned references to fields of packed structs
                at uncommon_codepoints detects uncommon Unicode codepoints in identifiers
                at unconditional_panic operation will cause a panic at runtime
                at unconditional_recursion functions that cannot return without calling themselves
                at unknown_crate_types unknown crate type found in `#[crate_type]` directive
                at unknown_lints    unrecognized lint attribute
                at unnameable_test_items detects an item that cannot be named being marked as `#[test_case]`
                at unreachable_code detects unreachable code paths
                at unreachable_patterns detects unreachable patterns
                at unreachable_pub  `pub` items not reachable from crate root
                at unsafe_code      usage of `unsafe` code
                at unsafe_op_in_unsafe_fn unsafe operations in unsafe functions without an explicit unsafe block are deprecated
                at unstable_features enabling unstable features (deprecated. do not use)
                at unstable_name_collisions detects name collision with an existing but unstable method
                at unused_allocation detects unnecessary allocations that can be eliminated
                at unused_assignments detect assignments that will never be read
                at unused_attributes detects attributes that were not used by the compiler
                at unused_braces    unnecessary braces around an expression
                at unused_comparisons comparisons made useless by limits of the types involved
                at unused_crate_dependencies crate dependencies that are never used
                at unused_doc_comments detects doc comments that aren't used by rustdoc
                at unused_extern_crates extern crates that are never used
                at unused_features  unused features found in crate-level `#[feature]` directives
                at unused_import_braces unnecessary braces around an imported item
                at unused_imports   imports that are never used
                at unused_labels    detects labels that are never used
                at unused_lifetimes detects lifetime parameters that are never used
                at unused_macros    detects macros that were not used
                at unused_must_use  unused result of a type flagged as `#[must_use]`
                at unused_mut       detect mut variables which don't need to be mutable
                at unused_parens    `if`, `match`, `while` and `return` do not need parentheses
                at unused_qualifications detects unnecessarily qualified names
                at unused_results   unused result of an expression in a statement
                at unused_unsafe    unnecessary use of an `unsafe` block
                at unused_variables detect variables which are not used in any way
                at variant_size_differences detects enums with widely varying variant sizes
                at warnings         mass-change the level for lints which produce warnings
                at where_clauses_object_safety checks the object safety of where clauses
                at while_true       suggest using `loop { }` instead of `while true { }`
        "#]],
        )
    }

    #[test]
    fn no_completion_for_incorrect_derive() {
        check(
            r#"
#[derive{<|>)]
struct Test {}
"#,
            expect![[r#""#]],
        )
    }

    #[test]
    fn derive_with_input_completion() {
        check(
            r#"
#[derive(serde::Serialize, PartialEq, <|>)]
struct Test {}
"#,
            expect![[r#"
                at Clone
                at Copy, Clone
                at Debug
                at Default
                at Eq
                at Hash
                at Ord, PartialOrd, Eq
                at PartialOrd
            "#]],
        )
    }

    #[test]
    fn test_attribute_completion() {
        check(
            r#"#[<|>]"#,
            expect![[r#"
                at allow(…)
                at cfg(…)
                at cfg_attr(…)
                at deny(…)
                at deprecated = "…"
                at derive(…)
                at doc = "…"
                at forbid(…)
                at ignore = "…"
                at inline(…)
                at link
                at link_name = "…"
                at macro_export
                at macro_use
                at must_use = "…"
                at no_mangle
                at non_exhaustive
                at path = "…"
                at proc_macro
                at proc_macro_attribute
                at proc_macro_derive(…)
                at repr(…)
                at should_panic(…)
                at target_feature = "…"
                at test
                at used
                at warn(…)
            "#]],
        )
    }

    #[test]
    fn test_attribute_completion_inside_nested_attr() {
        check(r#"#[cfg(<|>)]"#, expect![[]])
    }

    #[test]
    fn test_inner_attribute_completion() {
        check(
            r"#![<|>]",
            expect![[r#"
                at allow(…)
                at cfg(…)
                at cfg_attr(…)
                at deny(…)
                at deprecated = "…"
                at derive(…)
                at doc = "…"
                at feature(…)
                at forbid(…)
                at global_allocator
                at ignore = "…"
                at inline(…)
                at link
                at link_name = "…"
                at macro_export
                at macro_use
                at must_use = "…"
                at no_mangle
                at no_std
                at non_exhaustive
                at panic_handler
                at path = "…"
                at proc_macro
                at proc_macro_attribute
                at proc_macro_derive(…)
                at recursion_limit = …
                at repr(…)
                at should_panic(…)
                at target_feature = "…"
                at test
                at used
                at warn(…)
                at windows_subsystem = "…"
            "#]],
        );
    }
}
