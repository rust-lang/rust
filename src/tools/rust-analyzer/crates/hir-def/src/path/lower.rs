//! Transforms syntax into `Path` objects, ideally with accounting for hygiene

use crate::type_ref::ConstScalarOrPath;

use either::Either;
use hir_expand::name::{name, AsName};
use intern::Interned;
use syntax::ast::{self, AstNode, HasTypeBounds};

use super::AssociatedTypeBinding;
use crate::{
    body::LowerCtx,
    path::{GenericArg, GenericArgs, ModPath, Path, PathKind},
    type_ref::{LifetimeRef, TypeBound, TypeRef},
};

/// Converts an `ast::Path` to `Path`. Works with use trees.
/// It correctly handles `$crate` based path from macro call.
pub(super) fn lower_path(mut path: ast::Path, ctx: &LowerCtx<'_>) -> Option<Path> {
    let mut kind = PathKind::Plain;
    let mut type_anchor = None;
    let mut segments = Vec::new();
    let mut generic_args = Vec::new();
    let hygiene = ctx.hygiene();
    loop {
        let segment = path.segment()?;

        if segment.coloncolon_token().is_some() {
            kind = PathKind::Abs;
        }

        match segment.kind()? {
            ast::PathSegmentKind::Name(name_ref) => {
                // FIXME: this should just return name
                match hygiene.name_ref_to_name(ctx.db.upcast(), name_ref) {
                    Either::Left(name) => {
                        let args = segment
                            .generic_arg_list()
                            .and_then(|it| lower_generic_args(ctx, it))
                            .or_else(|| {
                                lower_generic_args_from_fn_path(
                                    ctx,
                                    segment.param_list(),
                                    segment.ret_type(),
                                )
                            })
                            .map(Interned::new);
                        segments.push(name);
                        generic_args.push(args)
                    }
                    Either::Right(crate_id) => {
                        kind = PathKind::DollarCrate(crate_id);
                        break;
                    }
                }
            }
            ast::PathSegmentKind::SelfTypeKw => {
                segments.push(name![Self]);
                generic_args.push(None)
            }
            ast::PathSegmentKind::Type { type_ref, trait_ref } => {
                assert!(path.qualifier().is_none()); // this can only occur at the first segment

                let self_type = TypeRef::from_ast(ctx, type_ref?);

                match trait_ref {
                    // <T>::foo
                    None => {
                        type_anchor = Some(Interned::new(self_type));
                        kind = PathKind::Plain;
                    }
                    // <T as Trait<A>>::Foo desugars to Trait<Self=T, A>::Foo
                    Some(trait_ref) => {
                        let Path { mod_path, generic_args: path_generic_args, .. } =
                            Path::from_src(trait_ref.path()?, ctx)?;
                        let num_segments = mod_path.segments().len();
                        kind = mod_path.kind;

                        segments.extend(mod_path.segments().iter().cloned().rev());
                        generic_args.extend(Vec::from(path_generic_args).into_iter().rev());

                        // Insert the type reference (T in the above example) as Self parameter for the trait
                        let last_segment =
                            generic_args.iter_mut().rev().nth(num_segments.saturating_sub(1))?;
                        let mut args_inner = match last_segment {
                            Some(it) => it.as_ref().clone(),
                            None => GenericArgs::empty(),
                        };
                        args_inner.has_self_type = true;
                        args_inner.args.insert(0, GenericArg::Type(self_type));
                        *last_segment = Some(Interned::new(args_inner));
                    }
                }
            }
            ast::PathSegmentKind::CrateKw => {
                kind = PathKind::Crate;
                break;
            }
            ast::PathSegmentKind::SelfKw => {
                // don't break out if `self` is the last segment of a path, this mean we got a
                // use tree like `foo::{self}` which we want to resolve as `foo`
                if !segments.is_empty() {
                    kind = PathKind::Super(0);
                    break;
                }
            }
            ast::PathSegmentKind::SuperKw => {
                let nested_super_count = if let PathKind::Super(n) = kind { n } else { 0 };
                kind = PathKind::Super(nested_super_count + 1);
            }
        }
        path = match qualifier(&path) {
            Some(it) => it,
            None => break,
        };
    }
    segments.reverse();
    generic_args.reverse();

    if segments.is_empty() && kind == PathKind::Plain && type_anchor.is_none() {
        // plain empty paths don't exist, this means we got a single `self` segment as our path
        kind = PathKind::Super(0);
    }

    // handle local_inner_macros :
    // Basically, even in rustc it is quite hacky:
    // https://github.com/rust-lang/rust/blob/614f273e9388ddd7804d5cbc80b8865068a3744e/src/librustc_resolve/macros.rs#L456
    // We follow what it did anyway :)
    if segments.len() == 1 && kind == PathKind::Plain {
        if let Some(_macro_call) = path.syntax().parent().and_then(ast::MacroCall::cast) {
            if let Some(crate_id) = hygiene.local_inner_macros(ctx.db.upcast(), path) {
                kind = PathKind::DollarCrate(crate_id);
            }
        }
    }

    let mod_path = Interned::new(ModPath::from_segments(kind, segments));
    return Some(Path { type_anchor, mod_path, generic_args: generic_args.into() });

    fn qualifier(path: &ast::Path) -> Option<ast::Path> {
        if let Some(q) = path.qualifier() {
            return Some(q);
        }
        // FIXME: this bottom up traversal is not too precise.
        // Should we handle do a top-down analysis, recording results?
        let use_tree_list = path.syntax().ancestors().find_map(ast::UseTreeList::cast)?;
        let use_tree = use_tree_list.parent_use_tree();
        use_tree.path()
    }
}

pub(super) fn lower_generic_args(
    lower_ctx: &LowerCtx<'_>,
    node: ast::GenericArgList,
) -> Option<GenericArgs> {
    let mut args = Vec::new();
    let mut bindings = Vec::new();
    for generic_arg in node.generic_args() {
        match generic_arg {
            ast::GenericArg::TypeArg(type_arg) => {
                let type_ref = TypeRef::from_ast_opt(lower_ctx, type_arg.ty());
                args.push(GenericArg::Type(type_ref));
            }
            ast::GenericArg::AssocTypeArg(assoc_type_arg) => {
                if let Some(name_ref) = assoc_type_arg.name_ref() {
                    let name = name_ref.as_name();
                    let args = assoc_type_arg
                        .generic_arg_list()
                        .and_then(|args| lower_generic_args(lower_ctx, args))
                        .map(Interned::new);
                    let type_ref = assoc_type_arg.ty().map(|it| TypeRef::from_ast(lower_ctx, it));
                    let bounds = if let Some(l) = assoc_type_arg.type_bound_list() {
                        l.bounds()
                            .map(|it| Interned::new(TypeBound::from_ast(lower_ctx, it)))
                            .collect()
                    } else {
                        Vec::new()
                    };
                    bindings.push(AssociatedTypeBinding { name, args, type_ref, bounds });
                }
            }
            ast::GenericArg::LifetimeArg(lifetime_arg) => {
                if let Some(lifetime) = lifetime_arg.lifetime() {
                    let lifetime_ref = LifetimeRef::new(&lifetime);
                    args.push(GenericArg::Lifetime(lifetime_ref))
                }
            }
            ast::GenericArg::ConstArg(arg) => {
                let arg = ConstScalarOrPath::from_expr_opt(arg.expr());
                args.push(GenericArg::Const(arg))
            }
        }
    }

    if args.is_empty() && bindings.is_empty() {
        return None;
    }
    Some(GenericArgs { args, has_self_type: false, bindings, desugared_from_fn: false })
}

/// Collect `GenericArgs` from the parts of a fn-like path, i.e. `Fn(X, Y)
/// -> Z` (which desugars to `Fn<(X, Y), Output=Z>`).
fn lower_generic_args_from_fn_path(
    ctx: &LowerCtx<'_>,
    params: Option<ast::ParamList>,
    ret_type: Option<ast::RetType>,
) -> Option<GenericArgs> {
    let mut args = Vec::new();
    let mut bindings = Vec::new();
    let params = params?;
    let mut param_types = Vec::new();
    for param in params.params() {
        let type_ref = TypeRef::from_ast_opt(ctx, param.ty());
        param_types.push(type_ref);
    }
    let arg = GenericArg::Type(TypeRef::Tuple(param_types));
    args.push(arg);
    if let Some(ret_type) = ret_type {
        let type_ref = TypeRef::from_ast_opt(ctx, ret_type.ty());
        bindings.push(AssociatedTypeBinding {
            name: name![Output],
            args: None,
            type_ref: Some(type_ref),
            bounds: Vec::new(),
        });
    } else {
        // -> ()
        let type_ref = TypeRef::Tuple(Vec::new());
        bindings.push(AssociatedTypeBinding {
            name: name![Output],
            args: None,
            type_ref: Some(type_ref),
            bounds: Vec::new(),
        });
    }
    Some(GenericArgs { args, has_self_type: false, bindings, desugared_from_fn: true })
}
