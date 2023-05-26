use hir::Semantics;
use ide_db::{
    base_db::{FileId, FileRange},
    defs::Definition,
    search::{SearchScope, UsageSearchResult},
    RootDatabase,
};
use syntax::{
    ast::{
        self, make::impl_trait_type, HasGenericParams, HasName, HasTypeBounds, Name, NameLike,
        PathType,
    },
    match_ast, ted, AstNode,
};
use text_edit::TextRange;

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: replace_named_generic_with_impl
//
// Replaces named generic with an `impl Trait` in function argument.
//
// ```
// fn new<P$0: AsRef<Path>>(location: P) -> Self {}
// ```
// ->
// ```
// fn new(location: impl AsRef<Path>) -> Self {}
// ```
pub(crate) fn replace_named_generic_with_impl(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    // finds `<P: AsRef<Path>>`
    let type_param = ctx.find_node_at_offset::<ast::TypeParam>()?;
    // returns `P`
    let type_param_name = type_param.name()?;

    // The list of type bounds / traits: `AsRef<Path>`
    let type_bound_list = type_param.type_bound_list()?;

    let fn_ = type_param.syntax().ancestors().find_map(ast::Fn::cast)?;
    let param_list_text_range = fn_.param_list()?.syntax().text_range();

    let type_param_hir_def = ctx.sema.to_def(&type_param)?;
    let type_param_def = Definition::GenericParam(hir::GenericParam::TypeParam(type_param_hir_def));

    // get all usage references for the type param
    let usage_refs = find_usages(&ctx.sema, &fn_, type_param_def, ctx.file_id());
    if usage_refs.is_empty() {
        return None;
    }

    // All usage references need to be valid (inside the function param list)
    if !check_valid_usages(&usage_refs, param_list_text_range) {
        return None;
    }

    let mut path_types_to_replace = Vec::new();
    for (_a, refs) in usage_refs.iter() {
        for usage_ref in refs {
            let param_node = find_path_type(&ctx.sema, &type_param_name, &usage_ref.name)?;
            path_types_to_replace.push(param_node);
        }
    }

    let target = type_param.syntax().text_range();

    acc.add(
        AssistId("replace_named_generic_with_impl", AssistKind::RefactorRewrite),
        "Replace named generic with impl trait",
        target,
        |edit| {
            let type_param = edit.make_mut(type_param);
            let fn_ = edit.make_mut(fn_);

            let path_types_to_replace = path_types_to_replace
                .into_iter()
                .map(|param| edit.make_mut(param))
                .collect::<Vec<_>>();

            // remove trait from generic param list
            if let Some(generic_params) = fn_.generic_param_list() {
                generic_params.remove_generic_param(ast::GenericParam::TypeParam(type_param));
                if generic_params.generic_params().count() == 0 {
                    ted::remove(generic_params.syntax());
                }
            }

            let new_bounds = impl_trait_type(type_bound_list);
            for path_type in path_types_to_replace.iter().rev() {
                ted::replace(path_type.syntax(), new_bounds.clone_for_update().syntax());
            }
        },
    )
}

fn find_path_type(
    sema: &Semantics<'_, RootDatabase>,
    type_param_name: &Name,
    param: &NameLike,
) -> Option<PathType> {
    let path_type =
        sema.ancestors_with_macros(param.syntax().clone()).find_map(ast::PathType::cast)?;

    // Ignore any path types that look like `P::Assoc`
    if path_type.path()?.as_single_name_ref()?.text() != type_param_name.text() {
        return None;
    }

    let ancestors = sema.ancestors_with_macros(path_type.syntax().clone());

    let mut in_generic_arg_list = false;
    let mut is_associated_type = false;

    // walking the ancestors checks them in a heuristic way until the `Fn` node is reached.
    for ancestor in ancestors {
        match_ast! {
            match ancestor {
                ast::PathSegment(ps) => {
                    match ps.kind()? {
                        ast::PathSegmentKind::Name(_name_ref) => (),
                        ast::PathSegmentKind::Type { .. } => return None,
                        _ => return None,
                    }
                },
                ast::GenericArgList(_) => {
                    in_generic_arg_list = true;
                },
                ast::AssocTypeArg(_) => {
                    is_associated_type = true;
                },
                ast::ImplTraitType(_) => {
                    if in_generic_arg_list && !is_associated_type {
                        return None;
                    }
                },
                ast::DynTraitType(_) => {
                    if !is_associated_type {
                        return None;
                    }
                },
                ast::Fn(_) => return Some(path_type),
                _ => (),
            }
        }
    }

    None
}

/// Returns all usage references for the given type parameter definition.
fn find_usages(
    sema: &Semantics<'_, RootDatabase>,
    fn_: &ast::Fn,
    type_param_def: Definition,
    file_id: FileId,
) -> UsageSearchResult {
    let file_range = FileRange { file_id, range: fn_.syntax().text_range() };
    type_param_def.usages(sema).in_scope(SearchScope::file_range(file_range)).all()
}

fn check_valid_usages(usages: &UsageSearchResult, param_list_range: TextRange) -> bool {
    usages
        .iter()
        .flat_map(|(_, usage_refs)| usage_refs)
        .all(|usage_ref| param_list_range.contains_range(usage_ref.range))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable};

    #[test]
    fn replace_generic_moves_into_function() {
        check_assist(
            replace_named_generic_with_impl,
            r#"fn new<T$0: ToString>(input: T) -> Self {}"#,
            r#"fn new(input: impl ToString) -> Self {}"#,
        );
    }

    #[test]
    fn replace_generic_with_inner_associated_type() {
        check_assist(
            replace_named_generic_with_impl,
            r#"fn new<P$0: AsRef<Path>>(input: P) -> Self {}"#,
            r#"fn new(input: impl AsRef<Path>) -> Self {}"#,
        );
    }

    #[test]
    fn replace_generic_trait_applies_to_all_matching_params() {
        check_assist(
            replace_named_generic_with_impl,
            r#"fn new<T$0: ToString>(a: T, b: T) -> Self {}"#,
            r#"fn new(a: impl ToString, b: impl ToString) -> Self {}"#,
        );
    }

    #[test]
    fn replace_generic_trait_applies_to_generic_arguments_in_params() {
        check_assist(
            replace_named_generic_with_impl,
            r#"
            fn foo<P$0: Trait>(
                _: P,
                _: Option<P>,
                _: Option<Option<P>>,
                _: impl Iterator<Item = P>,
                _: &dyn Iterator<Item = P>,
            ) {}
            "#,
            r#"
            fn foo(
                _: impl Trait,
                _: Option<impl Trait>,
                _: Option<Option<impl Trait>>,
                _: impl Iterator<Item = impl Trait>,
                _: &dyn Iterator<Item = impl Trait>,
            ) {}
            "#,
        );
    }

    #[test]
    fn replace_generic_not_applicable_when_one_param_type_is_invalid() {
        check_assist_not_applicable(
            replace_named_generic_with_impl,
            r#"
            fn foo<P$0: Trait>(
                _: i32,
                _: Option<P>,
                _: Option<Option<P>>,
                _: impl Iterator<Item = P>,
                _: &dyn Iterator<Item = P>,
                _: <P as Trait>::Assoc,
            ) {}
            "#,
        );
    }

    #[test]
    fn replace_generic_not_applicable_when_referenced_in_where_clause() {
        check_assist_not_applicable(
            replace_named_generic_with_impl,
            r#"fn foo<P$0: Trait, I>() where I: FromRef<P> {}"#,
        );
    }

    #[test]
    fn replace_generic_not_applicable_when_used_with_type_alias() {
        check_assist_not_applicable(
            replace_named_generic_with_impl,
            r#"fn foo<P$0: Trait>(p: <P as Trait>::Assoc) {}"#,
        );
    }

    #[test]
    fn replace_generic_not_applicable_when_used_as_argument_in_outer_trait_alias() {
        check_assist_not_applicable(
            replace_named_generic_with_impl,
            r#"fn foo<P$0: Trait>(_: <() as OtherTrait<P>>::Assoc) {}"#,
        );
    }

    #[test]
    fn replace_generic_not_applicable_with_inner_associated_type() {
        check_assist_not_applicable(
            replace_named_generic_with_impl,
            r#"fn foo<P$0: Trait>(_: P::Assoc) {}"#,
        );
    }

    #[test]
    fn replace_generic_not_applicable_when_passed_into_outer_impl_trait() {
        check_assist_not_applicable(
            replace_named_generic_with_impl,
            r#"fn foo<P$0: Trait>(_: impl OtherTrait<P>) {}"#,
        );
    }

    #[test]
    fn replace_generic_not_applicable_when_used_in_passed_function_parameter() {
        check_assist_not_applicable(
            replace_named_generic_with_impl,
            r#"fn foo<P$0: Trait>(_: &dyn Fn(P)) {}"#,
        );
    }

    #[test]
    fn replace_generic_with_multiple_generic_params() {
        check_assist(
            replace_named_generic_with_impl,
            r#"fn new<P: AsRef<Path>, T$0: ToString>(t: T, p: P) -> Self {}"#,
            r#"fn new<P: AsRef<Path>>(t: impl ToString, p: P) -> Self {}"#,
        );
        check_assist(
            replace_named_generic_with_impl,
            r#"fn new<T$0: ToString, P: AsRef<Path>>(t: T, p: P) -> Self {}"#,
            r#"fn new<P: AsRef<Path>>(t: impl ToString, p: P) -> Self {}"#,
        );
        check_assist(
            replace_named_generic_with_impl,
            r#"fn new<A: Send, B$0: ToString, C: Debug>(a: A, b: B, c: C) -> Self {}"#,
            r#"fn new<A: Send, C: Debug>(a: A, b: impl ToString, c: C) -> Self {}"#,
        );
    }

    #[test]
    fn replace_generic_with_multiple_trait_bounds() {
        check_assist(
            replace_named_generic_with_impl,
            r#"fn new<P$0: Send + Sync>(p: P) -> Self {}"#,
            r#"fn new(p: impl Send + Sync) -> Self {}"#,
        );
    }

    #[test]
    fn replace_generic_not_applicable_if_param_used_as_return_type() {
        check_assist_not_applicable(
            replace_named_generic_with_impl,
            r#"fn new<P$0: Send + Sync>(p: P) -> P {}"#,
        );
    }

    #[test]
    fn replace_generic_not_applicable_if_param_used_in_fn_body() {
        check_assist_not_applicable(
            replace_named_generic_with_impl,
            r#"fn new<P$0: ToString>(p: P) { let x: &dyn P = &O; }"#,
        );
    }

    #[test]
    fn replace_generic_ignores_another_function_with_same_param_type() {
        check_assist(
            replace_named_generic_with_impl,
            r#"
            fn new<P$0: Send + Sync>(p: P) {}
            fn hello<P: Debug>(p: P) { println!("{:?}", p); }
            "#,
            r#"
            fn new(p: impl Send + Sync) {}
            fn hello<P: Debug>(p: P) { println!("{:?}", p); }
            "#,
        );
    }
}
