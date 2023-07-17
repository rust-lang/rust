//! A lowering for `use`-paths (more generally, paths without angle-bracketed segments).

use std::{
    fmt::{self, Display as _},
    iter,
};

use crate::{
    db::ExpandDatabase,
    hygiene::Hygiene,
    name::{known, Name},
};
use base_db::CrateId;
use either::Either;
use smallvec::SmallVec;
use syntax::{ast, AstNode};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModPath {
    pub kind: PathKind,
    segments: SmallVec<[Name; 1]>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnescapedModPath<'a>(&'a ModPath);

impl<'a> UnescapedModPath<'a> {
    pub fn display(&'a self, db: &'a dyn crate::db::ExpandDatabase) -> impl fmt::Display + 'a {
        UnescapedDisplay { db, path: self }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PathKind {
    Plain,
    /// `self::` is `Super(0)`
    Super(u8),
    Crate,
    /// Absolute path (::foo)
    Abs,
    /// `$crate` from macro expansion
    DollarCrate(CrateId),
}

impl ModPath {
    pub fn from_src(
        db: &dyn ExpandDatabase,
        path: ast::Path,
        hygiene: &Hygiene,
    ) -> Option<ModPath> {
        convert_path(db, None, path, hygiene)
    }

    pub fn from_segments(kind: PathKind, segments: impl IntoIterator<Item = Name>) -> ModPath {
        let segments = segments.into_iter().collect();
        ModPath { kind, segments }
    }

    /// Creates a `ModPath` from a `PathKind`, with no extra path segments.
    pub const fn from_kind(kind: PathKind) -> ModPath {
        ModPath { kind, segments: SmallVec::new_const() }
    }

    pub fn segments(&self) -> &[Name] {
        &self.segments
    }

    pub fn push_segment(&mut self, segment: Name) {
        self.segments.push(segment);
    }

    pub fn pop_segment(&mut self) -> Option<Name> {
        self.segments.pop()
    }

    /// Returns the number of segments in the path (counting special segments like `$crate` and
    /// `super`).
    pub fn len(&self) -> usize {
        self.segments.len()
            + match self.kind {
                PathKind::Plain => 0,
                PathKind::Super(i) => i as usize,
                PathKind::Crate => 1,
                PathKind::Abs => 0,
                PathKind::DollarCrate(_) => 1,
            }
    }

    pub fn is_ident(&self) -> bool {
        self.as_ident().is_some()
    }

    pub fn is_self(&self) -> bool {
        self.kind == PathKind::Super(0) && self.segments.is_empty()
    }

    #[allow(non_snake_case)]
    pub fn is_Self(&self) -> bool {
        self.kind == PathKind::Plain
            && matches!(&*self.segments, [name] if *name == known::SELF_TYPE)
    }

    /// If this path is a single identifier, like `foo`, return its name.
    pub fn as_ident(&self) -> Option<&Name> {
        if self.kind != PathKind::Plain {
            return None;
        }

        match &*self.segments {
            [name] => Some(name),
            _ => None,
        }
    }

    pub fn unescaped(&self) -> UnescapedModPath<'_> {
        UnescapedModPath(self)
    }

    pub fn display<'a>(&'a self, db: &'a dyn crate::db::ExpandDatabase) -> impl fmt::Display + 'a {
        Display { db, path: self }
    }
}

struct Display<'a> {
    db: &'a dyn ExpandDatabase,
    path: &'a ModPath,
}

impl fmt::Display for Display<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        display_fmt_path(self.db, self.path, f, true)
    }
}

struct UnescapedDisplay<'a> {
    db: &'a dyn ExpandDatabase,
    path: &'a UnescapedModPath<'a>,
}

impl fmt::Display for UnescapedDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        display_fmt_path(self.db, self.path.0, f, false)
    }
}

impl From<Name> for ModPath {
    fn from(name: Name) -> ModPath {
        ModPath::from_segments(PathKind::Plain, iter::once(name))
    }
}
fn display_fmt_path(
    db: &dyn ExpandDatabase,
    path: &ModPath,
    f: &mut fmt::Formatter<'_>,
    escaped: bool,
) -> fmt::Result {
    let mut first_segment = true;
    let mut add_segment = |s| -> fmt::Result {
        if !first_segment {
            f.write_str("::")?;
        }
        first_segment = false;
        f.write_str(s)?;
        Ok(())
    };
    match path.kind {
        PathKind::Plain => {}
        PathKind::Super(0) => add_segment("self")?,
        PathKind::Super(n) => {
            for _ in 0..n {
                add_segment("super")?;
            }
        }
        PathKind::Crate => add_segment("crate")?,
        PathKind::Abs => add_segment("")?,
        PathKind::DollarCrate(_) => add_segment("$crate")?,
    }
    for segment in &path.segments {
        if !first_segment {
            f.write_str("::")?;
        }
        first_segment = false;
        if escaped {
            segment.display(db).fmt(f)?;
        } else {
            segment.unescaped().display(db).fmt(f)?;
        }
    }
    Ok(())
}

fn convert_path(
    db: &dyn ExpandDatabase,
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
            match hygiene.name_ref_to_name(db, name_ref) {
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
        ast::PathSegmentKind::SelfTypeKw => {
            if prefix.is_some() {
                return None;
            }
            ModPath::from_segments(PathKind::Plain, Some(known::SELF_TYPE))
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
            if let Some(crate_id) = hygiene.local_inner_macros(db, path) {
                mod_path.kind = PathKind::DollarCrate(crate_id);
            }
        }
    }

    Some(mod_path)
}

pub use crate::name as __name;

#[macro_export]
macro_rules! __known_path {
    (core::iter::IntoIterator) => {};
    (core::iter::Iterator) => {};
    (core::result::Result) => {};
    (core::option::Option) => {};
    (core::ops::Range) => {};
    (core::ops::RangeFrom) => {};
    (core::ops::RangeFull) => {};
    (core::ops::RangeTo) => {};
    (core::ops::RangeToInclusive) => {};
    (core::ops::RangeInclusive) => {};
    (core::future::Future) => {};
    (core::future::IntoFuture) => {};
    (core::ops::Try) => {};
    ($path:path) => {
        compile_error!("Please register your known path in the path module")
    };
}

#[macro_export]
macro_rules! __path {
    ($start:ident $(:: $seg:ident)*) => ({
        $crate::__known_path!($start $(:: $seg)*);
        $crate::mod_path::ModPath::from_segments($crate::mod_path::PathKind::Abs, vec![
            $crate::mod_path::__name![$start], $($crate::mod_path::__name![$seg],)*
        ])
    });
}

pub use crate::__path as path;
