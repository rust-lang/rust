//! Transforms syntax into `Path` objects, ideally with accounting for hygiene

#[cfg(test)]
mod tests;

use std::iter;

use crate::expr_store::{
    lower::{ExprCollector, generics::ImplTraitLowerFn},
    path::NormalPath,
};

use hir_expand::{
    mod_path::{ModPath, PathKind, resolve_crate_root},
    name::{AsName, Name},
};
use intern::{Interned, sym};
use syntax::{
    AstPtr,
    ast::{self, AstNode, HasGenericArgs},
};

use crate::{
    expr_store::path::{GenericArg, GenericArgs, Path},
    type_ref::TypeRef,
};

#[cfg(test)]
thread_local! {
    /// This is used to test `hir_segment_to_ast_segment()`. It's a hack, but it makes testing much easier.
    pub(super) static SEGMENT_LOWERING_MAP: std::cell::RefCell<rustc_hash::FxHashMap<ast::PathSegment, usize>> = std::cell::RefCell::default();
}

/// Converts an `ast::Path` to `Path`. Works with use trees.
/// It correctly handles `$crate` based path from macro call.
// If you modify the logic of the lowering, make sure to check if `hir_segment_to_ast_segment()`
// also needs an update.
pub(super) fn lower_path(
    collector: &mut ExprCollector<'_>,
    mut path: ast::Path,
    impl_trait_lower_fn: ImplTraitLowerFn<'_>,
) -> Option<Path> {
    let mut kind = PathKind::Plain;
    let mut type_anchor = None;
    let mut segments = Vec::new();
    let mut generic_args = Vec::new();
    #[cfg(test)]
    let mut ast_segments = Vec::new();
    #[cfg(test)]
    let mut ast_segments_offset = 0;
    #[allow(unused_mut)]
    let mut push_segment = |_segment: &ast::PathSegment, segments: &mut Vec<Name>, name| {
        #[cfg(test)]
        ast_segments.push(_segment.clone());
        segments.push(name);
    };
    loop {
        let Some(segment) = path.segment() else {
            segments.push(Name::missing());
            // We can end up here if for `path::`
            match qualifier(&path) {
                Some(it) => {
                    path = it;
                    continue;
                }
                None => break,
            }
        };

        if segment.coloncolon_token().is_some() {
            debug_assert!(path.qualifier().is_none()); // this can only occur at the first segment
            kind = PathKind::Abs;
        }

        match segment.kind()? {
            ast::PathSegmentKind::Name(name_ref) => {
                if name_ref.text() == "$crate" {
                    if path.qualifier().is_some() {
                        // FIXME: Report an error.
                        return None;
                    }
                    break kind = resolve_crate_root(
                        collector.db,
                        collector.expander.ctx_for_range(name_ref.syntax().text_range()),
                    )
                    .map(PathKind::DollarCrate)
                    .unwrap_or(PathKind::Crate);
                }
                let name = name_ref.as_name();
                let args = segment
                    .generic_arg_list()
                    .and_then(|it| collector.lower_generic_args(it, impl_trait_lower_fn))
                    .or_else(|| {
                        collector.lower_generic_args_from_fn_path(
                            segment.parenthesized_arg_list(),
                            segment.ret_type(),
                            impl_trait_lower_fn,
                        )
                    })
                    .or_else(|| {
                        segment.return_type_syntax().map(|_| GenericArgs::return_type_notation())
                    });
                if args.is_some() {
                    generic_args.resize(segments.len(), None);
                    generic_args.push(args);
                }
                push_segment(&segment, &mut segments, name);
            }
            ast::PathSegmentKind::SelfTypeKw => {
                push_segment(&segment, &mut segments, Name::new_symbol_root(sym::Self_));
            }
            ast::PathSegmentKind::Type { type_ref, trait_ref } => {
                debug_assert!(path.qualifier().is_none()); // this can only occur at the first segment

                let self_type = collector.lower_type_ref(type_ref?, impl_trait_lower_fn);

                match trait_ref {
                    // <T>::foo
                    None => {
                        type_anchor = Some(self_type);
                        kind = PathKind::Plain;
                    }
                    // <T as Trait<A>>::Foo desugars to Trait<Self=T, A>::Foo
                    Some(trait_ref) => {
                        let path = collector.lower_path(trait_ref.path()?, impl_trait_lower_fn)?;
                        // FIXME: Unnecessary clone
                        collector.alloc_type_ref(
                            TypeRef::Path(path.clone()),
                            AstPtr::new(&trait_ref).upcast(),
                        );
                        let mod_path = path.mod_path()?;
                        let path_generic_args = path.generic_args();
                        let num_segments = mod_path.segments().len();
                        kind = mod_path.kind;

                        segments.extend(mod_path.segments().iter().cloned().rev());
                        #[cfg(test)]
                        {
                            ast_segments_offset = mod_path.segments().len();
                        }
                        if let Some(path_generic_args) = path_generic_args {
                            generic_args.resize(segments.len() - num_segments, None);
                            generic_args.extend(Vec::from(path_generic_args).into_iter().rev());
                        } else {
                            generic_args.resize(segments.len(), None);
                        }

                        let self_type = GenericArg::Type(self_type);

                        // Insert the type reference (T in the above example) as Self parameter for the trait
                        let last_segment = generic_args.get_mut(segments.len() - num_segments)?;
                        *last_segment = Some(match last_segment.take() {
                            Some(it) => GenericArgs {
                                args: iter::once(self_type)
                                    .chain(it.args.iter().cloned())
                                    .collect(),
                                has_self_type: true,
                                ..it
                            },
                            None => GenericArgs {
                                args: Box::new([self_type]),
                                has_self_type: true,
                                ..GenericArgs::empty()
                            },
                        });
                    }
                }
            }
            ast::PathSegmentKind::CrateKw => {
                if path.qualifier().is_some() {
                    // FIXME: Report an error.
                    return None;
                }
                kind = PathKind::Crate;
                break;
            }
            ast::PathSegmentKind::SelfKw => {
                if path.qualifier().is_some() {
                    // FIXME: Report an error.
                    return None;
                }
                // don't break out if `self` is the last segment of a path, this mean we got a
                // use tree like `foo::{self}` which we want to resolve as `foo`
                if !segments.is_empty() {
                    kind = PathKind::SELF;
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
    if !generic_args.is_empty() || type_anchor.is_some() {
        generic_args.resize(segments.len(), None);
        generic_args.reverse();
    }

    if segments.is_empty() && kind == PathKind::Plain && type_anchor.is_none() {
        // plain empty paths don't exist, this means we got a single `self` segment as our path
        kind = PathKind::SELF;
    }

    // handle local_inner_macros :
    // Basically, even in rustc it is quite hacky:
    // https://github.com/rust-lang/rust/blob/614f273e9388ddd7804d5cbc80b8865068a3744e/src/librustc_resolve/macros.rs#L456
    // We follow what it did anyway :)
    if segments.len() == 1
        && kind == PathKind::Plain
        && let Some(_macro_call) = path.syntax().parent().and_then(ast::MacroCall::cast)
    {
        let syn_ctxt = collector.expander.ctx_for_range(path.segment()?.syntax().text_range());
        if let Some(macro_call_id) = syn_ctxt.outer_expn(collector.db)
            && collector.db.lookup_intern_macro_call(macro_call_id.into()).def.local_inner
        {
            kind = match resolve_crate_root(collector.db, syn_ctxt) {
                Some(crate_root) => PathKind::DollarCrate(crate_root),
                None => PathKind::Crate,
            }
        }
    }

    #[cfg(test)]
    {
        ast_segments.reverse();
        SEGMENT_LOWERING_MAP
            .with_borrow_mut(|map| map.extend(ast_segments.into_iter().zip(ast_segments_offset..)));
    }

    if let Some(last_segment_args @ Some(GenericArgs { has_self_type: true, .. })) =
        generic_args.last_mut()
    {
        // Well-formed code cannot have `<T as Trait>` without an associated item after,
        // and this causes panics in hir-ty lowering.
        *last_segment_args = None;
    }

    let mod_path = Interned::new(ModPath::from_segments(kind, segments));
    if type_anchor.is_none() && generic_args.is_empty() {
        return Some(Path::BarePath(mod_path));
    } else {
        return Some(Path::Normal(Box::new(NormalPath {
            type_anchor,
            mod_path,
            generic_args: generic_args.into_boxed_slice(),
        })));
    }

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

/// This function finds the AST segment that corresponds to the HIR segment
/// with index `segment_idx` on the path that is lowered from `path`.
pub fn hir_segment_to_ast_segment(path: &ast::Path, segment_idx: u32) -> Option<ast::PathSegment> {
    // Too tightly coupled to `lower_path()`, but unfortunately we cannot decouple them,
    // as keeping source maps for all paths segments will have a severe impact on memory usage.

    let mut segments = path.segments();
    if let Some(ast::PathSegmentKind::Type { trait_ref: Some(trait_ref), .. }) =
        segments.clone().next().and_then(|it| it.kind())
    {
        segments.next();
        return find_segment(trait_ref.path()?.segments().chain(segments), segment_idx);
    }
    return find_segment(segments, segment_idx);

    fn find_segment(
        segments: impl Iterator<Item = ast::PathSegment>,
        segment_idx: u32,
    ) -> Option<ast::PathSegment> {
        segments
            .filter(|segment| match segment.kind() {
                Some(
                    ast::PathSegmentKind::CrateKw
                    | ast::PathSegmentKind::SelfKw
                    | ast::PathSegmentKind::SuperKw
                    | ast::PathSegmentKind::Type { .. },
                )
                | None => false,
                Some(ast::PathSegmentKind::Name(name)) => name.text() != "$crate",
                Some(ast::PathSegmentKind::SelfTypeKw) => true,
            })
            .nth(segment_idx as usize)
    }
}
