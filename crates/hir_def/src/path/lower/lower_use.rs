//! Lowers a single complex use like `use foo::{bar, baz};` into a list of paths like
//! `foo::bar`, `foo::baz`;

use std::iter;

use either::Either;
use hir_expand::hygiene::Hygiene;
use syntax::{ast, AstNode};

use crate::{
    db::DefDatabase,
    path::{ModPath, PathKind},
};

pub(crate) fn convert_path(
    db: &dyn DefDatabase,
    prefix: Option<ModPath>,
    path: ast::Path,
    hygiene: &Hygiene,
) -> Option<ModPath> {
    let prefix = match path.qualifier() {
        Some(qual) => Some(convert_path(db, prefix, qual, hygiene)?),
        None => prefix,
    };

    let segment = path.segment()?;
    let mut mod_path = match segment.kind()? {
        ast::PathSegmentKind::Name(name_ref) => {
            match hygiene.name_ref_to_name(db.upcast(), name_ref) {
                Either::Left(name) => {
                    // no type args in use
                    let mut res = prefix.unwrap_or_else(|| {
                        ModPath::from_kind(
                            segment.coloncolon_token().map_or(PathKind::Plain, |_| PathKind::Abs),
                        )
                    });
                    res.segments.push(name);
                    res
                }
                Either::Right(crate_id) => {
                    return Some(ModPath::from_segments(
                        PathKind::DollarCrate(crate_id),
                        iter::empty(),
                    ))
                }
            }
        }
        ast::PathSegmentKind::CrateKw => {
            if prefix.is_some() {
                return None;
            }
            ModPath::from_segments(PathKind::Crate, iter::empty())
        }
        ast::PathSegmentKind::SelfKw => {
            if prefix.is_some() {
                return None;
            }
            ModPath::from_segments(PathKind::Super(0), iter::empty())
        }
        ast::PathSegmentKind::SuperKw => {
            let nested_super_count = match prefix.map(|p| p.kind) {
                Some(PathKind::Super(n)) => n,
                Some(_) => return None,
                None => 0,
            };

            ModPath::from_segments(PathKind::Super(nested_super_count + 1), iter::empty())
        }
        ast::PathSegmentKind::Type { .. } => {
            // not allowed in imports
            return None;
        }
    };

    // handle local_inner_macros :
    // Basically, even in rustc it is quite hacky:
    // https://github.com/rust-lang/rust/blob/614f273e9388ddd7804d5cbc80b8865068a3744e/src/librustc_resolve/macros.rs#L456
    // We follow what it did anyway :)
    if mod_path.segments.len() == 1 && mod_path.kind == PathKind::Plain {
        if let Some(_macro_call) = path.syntax().parent().and_then(ast::MacroCall::cast) {
            if let Some(crate_id) = hygiene.local_inner_macros(db.upcast(), path) {
                mod_path.kind = PathKind::DollarCrate(crate_id);
            }
        }
    }

    Some(mod_path)
}
