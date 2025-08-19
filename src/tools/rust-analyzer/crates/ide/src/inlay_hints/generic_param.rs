//! Implementation of inlay hints for generic parameters.
use either::Either;
use ide_db::{active_parameter::generic_def_for_node, famous_defs::FamousDefs};
use syntax::{
    AstNode,
    ast::{self, AnyHasGenericArgs, HasGenericArgs, HasName},
};

use crate::{
    InlayHint, InlayHintLabel, InlayHintsConfig, InlayKind,
    inlay_hints::{GenericParameterHints, param_name},
};

use super::param_name::is_argument_similar_to_param_name;

pub(crate) fn hints(
    acc: &mut Vec<InlayHint>,
    FamousDefs(sema, krate): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    node: AnyHasGenericArgs,
) -> Option<()> {
    let GenericParameterHints { type_hints, lifetime_hints, const_hints } =
        config.generic_parameter_hints;
    if !(type_hints || lifetime_hints || const_hints) {
        return None;
    }

    let generic_arg_list = node.generic_arg_list()?;

    let (generic_def, _, _, _) =
        generic_def_for_node(sema, &generic_arg_list, &node.syntax().first_token()?)?;

    let mut args = generic_arg_list.generic_args().peekable();
    let start_with_lifetime = matches!(args.peek()?, ast::GenericArg::LifetimeArg(_));
    let params = generic_def.params(sema.db).into_iter().filter(|p| {
        if let hir::GenericParam::TypeParam(it) = p
            && it.is_implicit(sema.db)
        {
            return false;
        }
        if !start_with_lifetime {
            return !matches!(p, hir::GenericParam::LifetimeParam(_));
        }
        true
    });

    let hints = params.zip(args).filter_map(|(param, arg)| {
        if matches!(arg, ast::GenericArg::AssocTypeArg(_)) {
            return None;
        }

        let allowed = match (param, &arg) {
            (hir::GenericParam::TypeParam(_), ast::GenericArg::TypeArg(_)) => type_hints,
            (hir::GenericParam::ConstParam(_), ast::GenericArg::ConstArg(_)) => const_hints,
            (hir::GenericParam::LifetimeParam(_), ast::GenericArg::LifetimeArg(_)) => {
                lifetime_hints
            }
            _ => false,
        };
        if !allowed {
            return None;
        }

        let param_name = param.name(sema.db);

        let should_hide = {
            let param_name = param_name.as_str();
            get_segment_representation(&arg).map_or(false, |seg| match seg {
                Either::Left(Either::Left(argument)) => {
                    is_argument_similar_to_param_name(&argument, param_name)
                }
                Either::Left(Either::Right(argument)) => argument
                    .segment()
                    .and_then(|it| it.name_ref())
                    .is_some_and(|it| it.text().eq_ignore_ascii_case(param_name)),
                Either::Right(lifetime) => lifetime.text().eq_ignore_ascii_case(param_name),
            })
        };

        if should_hide {
            return None;
        }

        let range = sema.original_range_opt(arg.syntax())?.range;

        let colon = if config.render_colons { ":" } else { "" };
        let label = InlayHintLabel::simple(
            format!("{}{colon}", param_name.display(sema.db, krate.edition(sema.db))),
            None,
            config.lazy_location_opt(|| {
                let source_syntax = match param {
                    hir::GenericParam::TypeParam(it) => {
                        sema.source(it.merge()).map(|it| it.value.syntax().clone())
                    }
                    hir::GenericParam::ConstParam(it) => {
                        let syntax = sema.source(it.merge())?.value.syntax().clone();
                        let const_param = ast::ConstParam::cast(syntax)?;
                        const_param.name().map(|it| it.syntax().clone())
                    }
                    hir::GenericParam::LifetimeParam(it) => {
                        sema.source(it).map(|it| it.value.syntax().clone())
                    }
                };
                let linked_location = source_syntax.and_then(|it| sema.original_range_opt(&it));
                linked_location.map(|frange| ide_db::FileRange {
                    file_id: frange.file_id.file_id(sema.db),
                    range: frange.range,
                })
            }),
        );

        Some(InlayHint {
            range,
            position: crate::InlayHintPosition::Before,
            pad_left: false,
            pad_right: true,
            kind: InlayKind::GenericParameter,
            label,
            text_edit: None,
            resolve_parent: Some(node.syntax().text_range()),
        })
    });

    acc.extend(hints);
    Some(())
}

fn get_segment_representation(
    arg: &ast::GenericArg,
) -> Option<Either<Either<Vec<ast::NameRef>, ast::Path>, ast::Lifetime>> {
    return match arg {
        ast::GenericArg::AssocTypeArg(_) => None,
        ast::GenericArg::ConstArg(const_arg) => {
            param_name::get_segment_representation(&const_arg.expr()?).map(Either::Left)
        }
        ast::GenericArg::LifetimeArg(lifetime_arg) => {
            let lifetime = lifetime_arg.lifetime()?;
            Some(Either::Right(lifetime))
        }
        ast::GenericArg::TypeArg(type_arg) => {
            let ty = type_arg.ty()?;
            type_path(&ty).map(Either::Right).map(Either::Left)
        }
    };

    fn type_path(ty: &ast::Type) -> Option<ast::Path> {
        match ty {
            ast::Type::ArrayType(it) => type_path(&it.ty()?),
            ast::Type::ForType(it) => type_path(&it.ty()?),
            ast::Type::ParenType(it) => type_path(&it.ty()?),
            ast::Type::PathType(path_type) => path_type.path(),
            ast::Type::PtrType(it) => type_path(&it.ty()?),
            ast::Type::RefType(it) => type_path(&it.ty()?),
            ast::Type::SliceType(it) => type_path(&it.ty()?),
            ast::Type::MacroType(macro_type) => macro_type.macro_call()?.path(),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        InlayHintsConfig,
        inlay_hints::{
            GenericParameterHints,
            tests::{DISABLED_CONFIG, check_with_config},
        },
    };

    #[track_caller]
    fn generic_param_name_hints_always(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
        check_with_config(
            InlayHintsConfig {
                generic_parameter_hints: GenericParameterHints {
                    type_hints: true,
                    lifetime_hints: true,
                    const_hints: true,
                },
                ..DISABLED_CONFIG
            },
            ra_fixture,
        );
    }

    #[track_caller]
    fn generic_param_name_hints_const_only(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
        check_with_config(
            InlayHintsConfig {
                generic_parameter_hints: GenericParameterHints {
                    type_hints: false,
                    lifetime_hints: false,
                    const_hints: true,
                },
                ..DISABLED_CONFIG
            },
            ra_fixture,
        );
    }

    #[test]
    fn type_only() {
        generic_param_name_hints_always(
            r#"
struct A<X, Y> {
    x: X,
    y: Y,
}

fn foo(a: A<usize,  u32>) {}
          //^^^^^ X ^^^ Y
"#,
        )
    }

    #[test]
    fn lifetime_and_type() {
        generic_param_name_hints_always(
            r#"
struct A<'a, X> {
    x: &'a X
}

fn foo<'b>(a: A<'b,  u32>) {}
              //^^ 'a^^^ X
"#,
        )
    }

    #[test]
    fn omit_lifetime() {
        generic_param_name_hints_always(
            r#"
struct A<'a, X> {
    x: &'a X
}

fn foo() {
    let x: i32 = 1;
    let a: A<i32> = A { x: &x };
          // ^^^ X
}
"#,
        )
    }

    #[test]
    fn const_only() {
        generic_param_name_hints_always(
            r#"
struct A<const X: usize, const Y: usize> {};

fn foo(a: A<12, 2>) {}
          //^^ X^ Y
"#,
        )
    }

    #[test]
    fn lifetime_and_type_and_const() {
        generic_param_name_hints_always(
            r#"
struct A<'a, X, const LEN: usize> {
    x: &'a [X; LEN],
}

fn foo<'b>(a: A<
    'b,
 // ^^ 'a
    u32,
 // ^^^ X
    3
 // ^ LEN
    >) {}
"#,
        )
    }

    #[test]
    fn const_only_config() {
        generic_param_name_hints_const_only(
            r#"
struct A<'a, X, const LEN: usize> {
    x: &'a [X; LEN],
}

fn foo<'b>(a: A<
    'b,
    u32,
    3
 // ^ LEN
    >) {}
"#,
        )
    }

    #[test]
    fn assoc_type() {
        generic_param_name_hints_always(
            r#"
trait Trait<T> {
    type Assoc1;
    type Assoc2;
}

fn foo() -> impl Trait<i32, Assoc1 = u32, Assoc2 = u32> {}
                    // ^^^ T
"#,
        )
    }

    #[test]
    fn hide_similar() {
        generic_param_name_hints_always(
            r#"
struct A<'a, X, const N: usize> {
    x: &'a [X; N],
}

const N: usize = 3;

mod m {
    type X = u32;
}

fn foo<'a>(a: A<'a, m::X, N>) {}
"#,
        )
    }

    #[test]
    fn mismatching_args() {
        generic_param_name_hints_always(
            r#"
struct A<X, const N: usize> {
    x: [X; N]
}

type InvalidType = A<3, i32>;
"#,
        )
    }
}
