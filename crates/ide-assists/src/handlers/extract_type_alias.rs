use either::Either;
use ide_db::syntax_helpers::node_ext::walk_ty;
use syntax::{
    ast::{self, edit::IndentLevel, make, AstNode, HasGenericParams, HasName},
    ted,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: extract_type_alias
//
// Extracts the selected type as a type alias.
//
// ```
// struct S {
//     field: $0(u8, u8, u8)$0,
// }
// ```
// ->
// ```
// type $0Type = (u8, u8, u8);
//
// struct S {
//     field: Type,
// }
// ```
pub(crate) fn extract_type_alias(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    if ctx.has_empty_selection() {
        return None;
    }

    let ty = ctx.find_node_at_range::<ast::Type>()?;
    let item = ty.syntax().ancestors().find_map(ast::Item::cast)?;
    let assoc_owner =
        item.syntax().ancestors().nth(2).and_then(Either::<ast::Trait, ast::Impl>::cast);
    let node = assoc_owner.as_ref().map_or_else(
        || item.syntax(),
        |impl_| impl_.as_ref().either(AstNode::syntax, AstNode::syntax),
    );
    let target = ty.syntax().text_range();

    acc.add(
        AssistId("extract_type_alias", AssistKind::RefactorExtract),
        "Extract type as type alias",
        target,
        |edit| {
            let node = edit.make_syntax_mut(node.clone());
            let target_ty = edit.make_mut(ty.clone());

            let mut known_generics = match item.generic_param_list() {
                Some(it) => it.generic_params().collect(),
                None => Vec::new(),
            };
            if let Some(it) = assoc_owner.as_ref().and_then(|it| match it {
                Either::Left(it) => it.generic_param_list(),
                Either::Right(it) => it.generic_param_list(),
            }) {
                known_generics.extend(it.generic_params());
            }
            let generics = collect_used_generics(&ty, &known_generics);
            let generic_params =
                generics.map(|it| make::generic_param_list(it.into_iter().cloned()));

            // Replace original type with the alias
            let ty_args = generic_params
                .as_ref()
                .map_or(String::new(), |it| it.to_generic_args().to_string());
            // FIXME: replace with a `ast::make` constructor
            let new_ty = make::ty(&format!("Type{ty_args}")).clone_for_update();
            ted::replace(target_ty.syntax(), new_ty.syntax());

            // Insert new alias
            let indent = IndentLevel::from_node(&node);
            let ty_alias = make::ty_alias("Type", generic_params, None, None, Some((ty, None)))
                .clone_for_update();
            ted::insert_all(
                ted::Position::before(node),
                vec![
                    ty_alias.syntax().clone().into(),
                    make::tokens::whitespace(&format!("\n\n{indent}")).into(),
                ],
            );

            if let Some(cap) = ctx.config.snippet_cap {
                if let Some(name) = ty_alias.name() {
                    edit.add_tabstop_before(cap, name);
                }
            }
        },
    )
}

fn collect_used_generics<'gp>(
    ty: &ast::Type,
    known_generics: &'gp [ast::GenericParam],
) -> Option<Vec<&'gp ast::GenericParam>> {
    // can't use a closure -> closure here cause lifetime inference fails for that
    fn find_lifetime(text: &str) -> impl Fn(&&ast::GenericParam) -> bool + '_ {
        move |gp: &&ast::GenericParam| match gp {
            ast::GenericParam::LifetimeParam(lp) => {
                lp.lifetime().map_or(false, |lt| lt.text() == text)
            }
            _ => false,
        }
    }

    let mut generics = Vec::new();
    walk_ty(ty, &mut |ty| {
        match ty {
            ast::Type::PathType(ty) => {
                if let Some(path) = ty.path() {
                    if let Some(name_ref) = path.as_single_name_ref() {
                        if let Some(param) = known_generics.iter().find(|gp| {
                            match gp {
                                ast::GenericParam::ConstParam(cp) => cp.name(),
                                ast::GenericParam::TypeParam(tp) => tp.name(),
                                _ => None,
                            }
                            .map_or(false, |n| n.text() == name_ref.text())
                        }) {
                            generics.push(param);
                        }
                    }
                    generics.extend(
                        path.segments()
                            .filter_map(|seg| seg.generic_arg_list())
                            .flat_map(|it| it.generic_args())
                            .filter_map(|it| match it {
                                ast::GenericArg::LifetimeArg(lt) => {
                                    let lt = lt.lifetime()?;
                                    known_generics.iter().find(find_lifetime(&lt.text()))
                                }
                                _ => None,
                            }),
                    );
                }
            }
            ast::Type::ImplTraitType(impl_ty) => {
                if let Some(it) = impl_ty.type_bound_list() {
                    generics.extend(
                        it.bounds()
                            .filter_map(|it| it.lifetime())
                            .filter_map(|lt| known_generics.iter().find(find_lifetime(&lt.text()))),
                    );
                }
            }
            ast::Type::DynTraitType(dyn_ty) => {
                if let Some(it) = dyn_ty.type_bound_list() {
                    generics.extend(
                        it.bounds()
                            .filter_map(|it| it.lifetime())
                            .filter_map(|lt| known_generics.iter().find(find_lifetime(&lt.text()))),
                    );
                }
            }
            ast::Type::RefType(ref_) => generics.extend(
                ref_.lifetime()
                    .and_then(|lt| known_generics.iter().find(find_lifetime(&lt.text()))),
            ),
            ast::Type::ArrayType(ar) => {
                if let Some(ast::Expr::PathExpr(p)) = ar.const_arg().and_then(|x| x.expr()) {
                    if let Some(path) = p.path() {
                        if let Some(name_ref) = path.as_single_name_ref() {
                            if let Some(param) = known_generics.iter().find(|gp| {
                                if let ast::GenericParam::ConstParam(cp) = gp {
                                    cp.name().map_or(false, |n| n.text() == name_ref.text())
                                } else {
                                    false
                                }
                            }) {
                                generics.push(param);
                            }
                        }
                    }
                }
            }
            _ => (),
        };
        false
    });
    // stable resort to lifetime, type, const
    generics.sort_by_key(|gp| match gp {
        ast::GenericParam::ConstParam(_) => 2,
        ast::GenericParam::LifetimeParam(_) => 0,
        ast::GenericParam::TypeParam(_) => 1,
    });

    Some(generics).filter(|it| it.len() > 0)
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn test_not_applicable_without_selection() {
        check_assist_not_applicable(
            extract_type_alias,
            r"
struct S {
    field: $0(u8, u8, u8),
}
            ",
        );
    }

    #[test]
    fn test_simple_types() {
        check_assist(
            extract_type_alias,
            r"
struct S {
    field: $0u8$0,
}
            ",
            r#"
type $0Type = u8;

struct S {
    field: Type,
}
            "#,
        );
    }

    #[test]
    fn test_generic_type_arg() {
        check_assist(
            extract_type_alias,
            r"
fn generic<T>() {}

fn f() {
    generic::<$0()$0>();
}
            ",
            r#"
fn generic<T>() {}

type $0Type = ();

fn f() {
    generic::<Type>();
}
            "#,
        );
    }

    #[test]
    fn test_inner_type_arg() {
        check_assist(
            extract_type_alias,
            r"
struct Vec<T> {}
struct S {
    v: Vec<Vec<$0Vec<u8>$0>>,
}
            ",
            r#"
struct Vec<T> {}
type $0Type = Vec<u8>;

struct S {
    v: Vec<Vec<Type>>,
}
            "#,
        );
    }

    #[test]
    fn test_extract_inner_type() {
        check_assist(
            extract_type_alias,
            r"
struct S {
    field: ($0u8$0,),
}
            ",
            r#"
type $0Type = u8;

struct S {
    field: (Type,),
}
            "#,
        );
    }

    #[test]
    fn extract_from_impl_or_trait() {
        // When invoked in an impl/trait, extracted type alias should be placed next to the
        // impl/trait, not inside.
        check_assist(
            extract_type_alias,
            r#"
impl S {
    fn f() -> $0(u8, u8)$0 {}
}
            "#,
            r#"
type $0Type = (u8, u8);

impl S {
    fn f() -> Type {}
}
            "#,
        );
        check_assist(
            extract_type_alias,
            r#"
trait Tr {
    fn f() -> $0(u8, u8)$0 {}
}
            "#,
            r#"
type $0Type = (u8, u8);

trait Tr {
    fn f() -> Type {}
}
            "#,
        );
    }

    #[test]
    fn indentation() {
        check_assist(
            extract_type_alias,
            r#"
mod m {
    fn f() -> $0u8$0 {}
}
            "#,
            r#"
mod m {
    type $0Type = u8;

    fn f() -> Type {}
}
            "#,
        );
    }

    #[test]
    fn generics() {
        check_assist(
            extract_type_alias,
            r#"
struct Struct<const C: usize>;
impl<'outer, Outer, const OUTER: usize> () {
    fn func<'inner, Inner, const INNER: usize>(_: $0&(Struct<INNER>, Struct<OUTER>, Outer, &'inner (), Inner, &'outer ())$0) {}
}
"#,
            r#"
struct Struct<const C: usize>;
type $0Type<'inner, 'outer, Outer, Inner, const INNER: usize, const OUTER: usize> = &(Struct<INNER>, Struct<OUTER>, Outer, &'inner (), Inner, &'outer ());

impl<'outer, Outer, const OUTER: usize> () {
    fn func<'inner, Inner, const INNER: usize>(_: Type<'inner, 'outer, Outer, Inner, INNER, OUTER>) {}
}
"#,
        );
    }

    #[test]
    fn issue_11197() {
        check_assist(
            extract_type_alias,
            r#"
struct Foo<T, const N: usize>
where
    [T; N]: Sized,
{
    arr: $0[T; N]$0,
}
            "#,
            r#"
type $0Type<T, const N: usize> = [T; N];

struct Foo<T, const N: usize>
where
    [T; N]: Sized,
{
    arr: Type<T, N>,
}
            "#,
        );
    }
}
