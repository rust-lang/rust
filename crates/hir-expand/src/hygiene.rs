//! This modules handles hygiene information.
//!
//! Specifically, `ast` + `Hygiene` allows you to create a `Name`. Note that, at
//! this moment, this is horribly incomplete and handles only `$crate`.
use base_db::{span::SyntaxContextId, CrateId};
use either::Either;
use syntax::{
    ast::{self},
    TextRange,
};
use triomphe::Arc;

use crate::{
    db::ExpandDatabase,
    name::{AsName, Name},
    HirFileId, InFile,
};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct SyntaxContextData {
    // FIXME: This might only need to be Option<MacroCallId>?
    outer_expn: HirFileId,
    outer_transparency: Transparency,
    parent: SyntaxContextId,
    /// This context, but with all transparent and semi-transparent expansions filtered away.
    opaque: SyntaxContextId,
    /// This context, but with all transparent expansions filtered away.
    opaque_and_semitransparent: SyntaxContextId,
    /// Name of the crate to which `$crate` with this context would resolve.
    dollar_crate_name: Name,
}

/// A property of a macro expansion that determines how identifiers
/// produced by that expansion are resolved.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Hash, Debug)]
pub enum Transparency {
    /// Identifier produced by a transparent expansion is always resolved at call-site.
    /// Call-site spans in procedural macros, hygiene opt-out in `macro` should use this.
    Transparent,
    /// Identifier produced by a semi-transparent expansion may be resolved
    /// either at call-site or at definition-site.
    /// If it's a local variable, label or `$crate` then it's resolved at def-site.
    /// Otherwise it's resolved at call-site.
    /// `macro_rules` macros behave like this, built-in macros currently behave like this too,
    /// but that's an implementation detail.
    SemiTransparent,
    /// Identifier produced by an opaque expansion is always resolved at definition-site.
    /// Def-site spans in procedural macros, identifiers from `macro` by default use this.
    Opaque,
}

pub(super) fn apply_mark(
    _db: &dyn ExpandDatabase,
    _ctxt: SyntaxContextData,
    _file_id: HirFileId,
    _transparency: Transparency,
) -> SyntaxContextId {
    _db.intern_syntax_context(_ctxt)
}

// pub(super) fn with_ctxt_from_mark(db: &ExpandDatabase, file_id: HirFileId) {
//     self.with_ctxt_from_mark(expn_id, Transparency::Transparent)
// }
// pub(super) fn with_call_site_ctxt(db: &ExpandDatabase, file_id: HirFileId) {
//     self.with_ctxt_from_mark(expn_id, Transparency::Transparent)
// }

#[derive(Clone, Debug)]
pub struct Hygiene {}

impl Hygiene {
    pub fn new(_: &dyn ExpandDatabase, _: HirFileId) -> Hygiene {
        Hygiene {}
    }

    pub fn new_unhygienic() -> Hygiene {
        Hygiene {}
    }

    // FIXME: this should just return name
    pub fn name_ref_to_name(
        &self,
        _: &dyn ExpandDatabase,
        name_ref: ast::NameRef,
    ) -> Either<Name, CrateId> {
        Either::Left(name_ref.as_name())
    }

    pub fn local_inner_macros(&self, _: &dyn ExpandDatabase, _: ast::Path) -> Option<CrateId> {
        None
    }
}

#[derive(Clone, Debug)]
struct HygieneFrames(Arc<HygieneFrame>);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HygieneFrame {}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HygieneInfo {}

impl HygieneInfo {
    fn _map_ident_up(&self, _: &dyn ExpandDatabase, _: TextRange) -> Option<InFile<TextRange>> {
        None
    }
}

impl HygieneFrame {
    pub(crate) fn new(_: &dyn ExpandDatabase, _: HirFileId) -> HygieneFrame {
        HygieneFrame {}
    }
}
