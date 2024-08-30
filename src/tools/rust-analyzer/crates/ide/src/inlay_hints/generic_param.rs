//! Implementation of inlay hints for generic parameters.
use ide_db::{active_parameter::generic_def_for_node, RootDatabase};
use syntax::{
    ast::{self, AnyHasGenericArgs, HasGenericArgs, HasName},
    AstNode,
};

use crate::{inlay_hints::GenericParameterHints, InlayHint, InlayHintsConfig, InlayKind};

use super::param_name::{is_argument_similar_to_param_name, render_label};

pub(crate) fn hints(
    acc: &mut Vec<InlayHint>,
    sema: &hir::Semantics<'_, RootDatabase>,
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
        if let hir::GenericParam::TypeParam(it) = p {
            if it.is_implicit(sema.db) {
                return false;
            }
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

        let name = param.name(sema.db);
        let param_name = name.as_str();

        let should_hide = {
            let argument = get_string_representation(&arg)?;
            is_argument_similar_to_param_name(&argument, param_name)
        };

        if should_hide {
            return None;
        }

        let range = sema.original_range_opt(arg.syntax())?.range;

        let source_syntax = match param {
            hir::GenericParam::TypeParam(it) => {
                if !type_hints || !matches!(arg, ast::GenericArg::TypeArg(_)) {
                    return None;
                }
                sema.source(it.merge())?.value.syntax().clone()
            }
            hir::GenericParam::ConstParam(it) => {
                if !const_hints || !matches!(arg, ast::GenericArg::ConstArg(_)) {
                    return None;
                }
                let syntax = sema.source(it.merge())?.value.syntax().clone();
                let const_param = ast::ConstParam::cast(syntax)?;
                const_param.name()?.syntax().clone()
            }
            hir::GenericParam::LifetimeParam(it) => {
                if !lifetime_hints || !matches!(arg, ast::GenericArg::LifetimeArg(_)) {
                    return None;
                }
                sema.source(it)?.value.syntax().clone()
            }
        };
        let linked_location = sema.original_range_opt(&source_syntax);
        let label = render_label(param_name, config, linked_location);

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

fn get_string_representation(arg: &ast::GenericArg) -> Option<String> {
    return match arg {
        ast::GenericArg::AssocTypeArg(_) => None,
        ast::GenericArg::ConstArg(const_arg) => Some(const_arg.to_string()),
        ast::GenericArg::LifetimeArg(lifetime_arg) => {
            let lifetime = lifetime_arg.lifetime()?;
            Some(lifetime.to_string())
        }
        ast::GenericArg::TypeArg(type_arg) => {
            let ty = type_arg.ty()?;
            Some(
                type_path_segment(&ty)
                    .map_or_else(|| type_arg.to_string(), |segment| segment.to_string()),
            )
        }
    };

    fn type_path_segment(ty: &ast::Type) -> Option<ast::PathSegment> {
        match ty {
            ast::Type::ArrayType(it) => type_path_segment(&it.ty()?),
            ast::Type::ForType(it) => type_path_segment(&it.ty()?),
            ast::Type::ParenType(it) => type_path_segment(&it.ty()?),
            ast::Type::PathType(path_type) => path_type.path()?.segment(),
            ast::Type::PtrType(it) => type_path_segment(&it.ty()?),
            ast::Type::RefType(it) => type_path_segment(&it.ty()?),
            ast::Type::SliceType(it) => type_path_segment(&it.ty()?),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        inlay_hints::{
            tests::{check_with_config, DISABLED_CONFIG},
            GenericParameterHints,
        },
        InlayHintsConfig,
    };

    #[track_caller]
    fn generic_param_name_hints_always(ra_fixture: &str) {
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
    fn generic_param_name_hints_const_only(ra_fixture: &str) {
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
