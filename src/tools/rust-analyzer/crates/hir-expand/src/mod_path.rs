//! A lowering for `use`-paths (more generally, paths without angle-bracketed segments).

use std::{
    fmt::{self, Display as _},
    iter,
};

use crate::{
    db::ExpandDatabase,
    hygiene::Transparency,
    name::{AsName, Name},
    tt,
};
use base_db::Crate;
use intern::sym;
use smallvec::SmallVec;
use span::{Edition, SyntaxContext};
use syntax::{AstNode, ast};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModPath {
    pub kind: PathKind,
    segments: SmallVec<[Name; 1]>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PathKind {
    Plain,
    /// `self::` is `Super(0)`
    Super(u8),
    Crate,
    /// Absolute path (::foo)
    Abs,
    // FIXME: Can we remove this somehow?
    /// `$crate` from macro expansion
    DollarCrate(Crate),
}

impl PathKind {
    pub const SELF: PathKind = PathKind::Super(0);
}

impl ModPath {
    pub fn from_src(
        db: &dyn ExpandDatabase,
        path: ast::Path,
        span_for_range: &mut dyn FnMut(::tt::TextRange) -> SyntaxContext,
    ) -> Option<ModPath> {
        convert_path(db, path, span_for_range)
    }

    pub fn from_tt(db: &dyn ExpandDatabase, tt: tt::TokenTreesView<'_>) -> Option<ModPath> {
        convert_path_tt(db, tt)
    }

    pub fn from_segments(kind: PathKind, segments: impl IntoIterator<Item = Name>) -> ModPath {
        let mut segments: SmallVec<_> = segments.into_iter().collect();
        segments.shrink_to_fit();
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

    pub fn textual_len(&self) -> usize {
        let base = match self.kind {
            PathKind::Plain => 0,
            PathKind::SELF => "self".len(),
            PathKind::Super(i) => "super".len() * i as usize,
            PathKind::Crate => "crate".len(),
            PathKind::Abs => 0,
            PathKind::DollarCrate(_) => "$crate".len(),
        };
        self.segments().iter().map(|segment| segment.as_str().len()).fold(base, core::ops::Add::add)
    }

    pub fn is_ident(&self) -> bool {
        self.as_ident().is_some()
    }

    pub fn is_self(&self) -> bool {
        self.kind == PathKind::SELF && self.segments.is_empty()
    }

    #[allow(non_snake_case)]
    pub fn is_Self(&self) -> bool {
        self.kind == PathKind::Plain && matches!(&*self.segments, [name] if *name == sym::Self_)
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
    pub fn display_verbatim<'a>(
        &'a self,
        db: &'a dyn crate::db::ExpandDatabase,
    ) -> impl fmt::Display + 'a {
        Display { db, path: self, edition: None }
    }

    pub fn display<'a>(
        &'a self,
        db: &'a dyn crate::db::ExpandDatabase,
        edition: Edition,
    ) -> impl fmt::Display + 'a {
        Display { db, path: self, edition: Some(edition) }
    }
}

impl Extend<Name> for ModPath {
    fn extend<T: IntoIterator<Item = Name>>(&mut self, iter: T) {
        self.segments.extend(iter);
    }
}

struct Display<'a> {
    db: &'a dyn ExpandDatabase,
    path: &'a ModPath,
    edition: Option<Edition>,
}

impl fmt::Display for Display<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        display_fmt_path(self.db, self.path, f, self.edition)
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
    edition: Option<Edition>,
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
        PathKind::SELF => add_segment("self")?,
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
        match edition {
            Some(edition) => segment.display(db, edition).fmt(f)?,
            None => fmt::Display::fmt(segment.as_str(), f)?,
        };
    }
    Ok(())
}

fn convert_path(
    db: &dyn ExpandDatabase,
    path: ast::Path,
    span_for_range: &mut dyn FnMut(::tt::TextRange) -> SyntaxContext,
) -> Option<ModPath> {
    let mut segments = path.segments();

    let segment = &segments.next()?;
    let handle_super_kw = &mut |init_deg| {
        let mut deg = init_deg;
        let mut next_segment = None;
        for segment in segments.by_ref() {
            match segment.kind()? {
                ast::PathSegmentKind::SuperKw => deg += 1,
                ast::PathSegmentKind::Name(name) => {
                    next_segment = Some(name.as_name());
                    break;
                }
                ast::PathSegmentKind::Type { .. }
                | ast::PathSegmentKind::SelfTypeKw
                | ast::PathSegmentKind::SelfKw
                | ast::PathSegmentKind::CrateKw => return None,
            }
        }

        Some(ModPath::from_segments(PathKind::Super(deg), next_segment))
    };

    let mut mod_path = match segment.kind()? {
        ast::PathSegmentKind::Name(name_ref) => {
            if name_ref.text() == "$crate" {
                ModPath::from_kind(
                    resolve_crate_root(db, span_for_range(name_ref.syntax().text_range()))
                        .map(PathKind::DollarCrate)
                        .unwrap_or(PathKind::Crate),
                )
            } else {
                let mut res = ModPath::from_kind(
                    segment.coloncolon_token().map_or(PathKind::Plain, |_| PathKind::Abs),
                );
                res.segments.push(name_ref.as_name());
                res
            }
        }
        ast::PathSegmentKind::SelfTypeKw => {
            ModPath::from_segments(PathKind::Plain, Some(Name::new_symbol_root(sym::Self_)))
        }
        ast::PathSegmentKind::CrateKw => ModPath::from_segments(PathKind::Crate, iter::empty()),
        ast::PathSegmentKind::SelfKw => handle_super_kw(0)?,
        ast::PathSegmentKind::SuperKw => handle_super_kw(1)?,
        ast::PathSegmentKind::Type { .. } => {
            // not allowed in imports
            return None;
        }
    };

    for segment in segments {
        let name = match segment.kind()? {
            ast::PathSegmentKind::Name(name) => name.as_name(),
            _ => return None,
        };
        mod_path.segments.push(name);
    }

    // handle local_inner_macros :
    // Basically, even in rustc it is quite hacky:
    // https://github.com/rust-lang/rust/blob/614f273e9388ddd7804d5cbc80b8865068a3744e/src/librustc_resolve/macros.rs#L456
    // We follow what it did anyway :)
    if mod_path.segments.len() == 1
        && mod_path.kind == PathKind::Plain
        && let Some(_macro_call) = path.syntax().parent().and_then(ast::MacroCall::cast)
    {
        let syn_ctx = span_for_range(segment.syntax().text_range());
        if let Some(macro_call_id) = syn_ctx.outer_expn(db)
            && db.lookup_intern_macro_call(macro_call_id.into()).def.local_inner
        {
            mod_path.kind = match resolve_crate_root(db, syn_ctx) {
                Some(crate_root) => PathKind::DollarCrate(crate_root),
                None => PathKind::Crate,
            }
        }
    }

    Some(mod_path)
}

fn convert_path_tt(db: &dyn ExpandDatabase, tt: tt::TokenTreesView<'_>) -> Option<ModPath> {
    let mut leaves = tt.iter().filter_map(|tt| match tt {
        tt::TtElement::Leaf(leaf) => Some(leaf),
        tt::TtElement::Subtree(..) => None,
    });
    let mut segments = smallvec::smallvec![];
    let kind = match leaves.next()? {
        tt::Leaf::Punct(tt::Punct { char: ':', .. }) => match leaves.next()? {
            tt::Leaf::Punct(tt::Punct { char: ':', .. }) => PathKind::Abs,
            _ => return None,
        },
        tt::Leaf::Ident(tt::Ident { sym: text, span, .. }) if *text == sym::dollar_crate => {
            resolve_crate_root(db, span.ctx).map(PathKind::DollarCrate).unwrap_or(PathKind::Crate)
        }
        tt::Leaf::Ident(tt::Ident { sym: text, .. }) if *text == sym::self_ => PathKind::SELF,
        tt::Leaf::Ident(tt::Ident { sym: text, .. }) if *text == sym::super_ => {
            let mut deg = 1;
            while let Some(tt::Leaf::Ident(tt::Ident { sym: text, span, is_raw: _ })) =
                leaves.next()
            {
                if *text != sym::super_ {
                    segments.push(Name::new_symbol(text.clone(), span.ctx));
                    break;
                }
                deg += 1;
            }
            PathKind::Super(deg)
        }
        tt::Leaf::Ident(tt::Ident { sym: text, .. }) if *text == sym::crate_ => PathKind::Crate,
        tt::Leaf::Ident(ident) => {
            segments.push(Name::new_symbol(ident.sym.clone(), ident.span.ctx));
            PathKind::Plain
        }
        _ => return None,
    };
    segments.extend(leaves.filter_map(|leaf| match leaf {
        ::tt::Leaf::Ident(ident) => Some(Name::new_symbol(ident.sym.clone(), ident.span.ctx)),
        _ => None,
    }));
    Some(ModPath { kind, segments })
}

pub fn resolve_crate_root(db: &dyn ExpandDatabase, mut ctxt: SyntaxContext) -> Option<Crate> {
    // When resolving `$crate` from a `macro_rules!` invoked in a `macro`,
    // we don't want to pretend that the `macro_rules!` definition is in the `macro`
    // as described in `SyntaxContextId::apply_mark`, so we ignore prepended opaque marks.
    // FIXME: This is only a guess and it doesn't work correctly for `macro_rules!`
    // definitions actually produced by `macro` and `macro` definitions produced by
    // `macro_rules!`, but at least such configurations are not stable yet.
    ctxt = ctxt.normalize_to_macro_rules(db);
    let mut iter = ctxt.marks_rev(db).peekable();
    let mut result_mark = None;
    // Find the last opaque mark from the end if it exists.
    while let Some(&(mark, Transparency::Opaque)) = iter.peek() {
        result_mark = Some(mark);
        iter.next();
    }
    // Then find the last semi-transparent mark from the end if it exists.
    while let Some((mark, Transparency::SemiTransparent)) = iter.next() {
        result_mark = Some(mark);
    }

    result_mark.map(|call| db.lookup_intern_macro_call(call.into()).def.krate)
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
    (core::fmt::Debug) => {};
    (std::fmt::format) => {};
    (core::ops::Try) => {};
    (core::convert::From) => {};
    (core::convert::TryFrom) => {};
    (core::str::FromStr) => {};
    ($path:path) => {
        compile_error!("Please register your known path in the path module")
    };
}

#[macro_export]
macro_rules! __path {
    ($start:ident $(:: $seg:ident)*) => ({
        $crate::__known_path!($start $(:: $seg)*);
        $crate::mod_path::ModPath::from_segments($crate::mod_path::PathKind::Abs, vec![
            $crate::name::Name::new_symbol_root($crate::intern::sym::$start.clone()), $($crate::name::Name::new_symbol_root($crate::intern::sym::$seg.clone()),)*
        ])
    });
}

pub use crate::__path as path;

#[macro_export]
macro_rules! __tool_path {
    ($start:ident $(:: $seg:ident)*) => ({
        $crate::mod_path::ModPath::from_segments($crate::mod_path::PathKind::Plain, vec![
            $crate::name::Name::new_symbol_root($crate::intern::sym::rust_analyzer), $crate::name::Name::new_symbol_root($crate::intern::sym::$start.clone()), $($crate::name::Name::new_symbol_root($crate::intern::sym::$seg.clone()),)*
        ])
    });
}

pub use crate::__tool_path as tool_path;
