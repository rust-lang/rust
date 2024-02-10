//! This modules handles hygiene information.
//!
//! Specifically, `ast` + `Hygiene` allows you to create a `Name`. Note that, at
//! this moment, this is horribly incomplete and handles only `$crate`.

// FIXME: Consider moving this into the span crate.

use std::iter;

use base_db::salsa::{self, InternValue};
use span::{MacroCallId, Span, SyntaxContextId};

use crate::db::{ExpandDatabase, InternSyntaxContextQuery};

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct SyntaxContextData {
    pub outer_expn: Option<MacroCallId>,
    pub outer_transparency: Transparency,
    pub parent: SyntaxContextId,
    /// This context, but with all transparent and semi-transparent expansions filtered away.
    pub opaque: SyntaxContextId,
    /// This context, but with all transparent expansions filtered away.
    pub opaque_and_semitransparent: SyntaxContextId,
}

impl InternValue for SyntaxContextData {
    type Key = (SyntaxContextId, Option<MacroCallId>, Transparency);

    fn into_key(&self) -> Self::Key {
        (self.parent, self.outer_expn, self.outer_transparency)
    }
}

impl std::fmt::Debug for SyntaxContextData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyntaxContextData")
            .field("outer_expn", &self.outer_expn)
            .field("outer_transparency", &self.outer_transparency)
            .field("parent", &self.parent)
            .field("opaque", &self.opaque)
            .field("opaque_and_semitransparent", &self.opaque_and_semitransparent)
            .finish()
    }
}

impl SyntaxContextData {
    pub fn root() -> Self {
        SyntaxContextData {
            outer_expn: None,
            outer_transparency: Transparency::Opaque,
            parent: SyntaxContextId::ROOT,
            opaque: SyntaxContextId::ROOT,
            opaque_and_semitransparent: SyntaxContextId::ROOT,
        }
    }

    pub fn fancy_debug(
        self,
        self_id: SyntaxContextId,
        db: &dyn ExpandDatabase,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "#{self_id} parent: #{}, outer_mark: (", self.parent)?;
        match self.outer_expn {
            Some(id) => {
                write!(f, "{:?}::{{{{expn{:?}}}}}", db.lookup_intern_macro_call(id).krate, id)?
            }
            None => write!(f, "root")?,
        }
        write!(f, ", {:?})", self.outer_transparency)
    }
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

pub fn span_with_def_site_ctxt(db: &dyn ExpandDatabase, span: Span, expn_id: MacroCallId) -> Span {
    span_with_ctxt_from_mark(db, span, expn_id, Transparency::Opaque)
}

pub fn span_with_call_site_ctxt(db: &dyn ExpandDatabase, span: Span, expn_id: MacroCallId) -> Span {
    span_with_ctxt_from_mark(db, span, expn_id, Transparency::Transparent)
}

pub fn span_with_mixed_site_ctxt(
    db: &dyn ExpandDatabase,
    span: Span,
    expn_id: MacroCallId,
) -> Span {
    span_with_ctxt_from_mark(db, span, expn_id, Transparency::SemiTransparent)
}

fn span_with_ctxt_from_mark(
    db: &dyn ExpandDatabase,
    span: Span,
    expn_id: MacroCallId,
    transparency: Transparency,
) -> Span {
    Span { ctx: apply_mark(db, SyntaxContextId::ROOT, expn_id, transparency), ..span }
}

pub(super) fn apply_mark(
    db: &dyn ExpandDatabase,
    ctxt: SyntaxContextId,
    call_id: MacroCallId,
    transparency: Transparency,
) -> SyntaxContextId {
    if transparency == Transparency::Opaque {
        return apply_mark_internal(db, ctxt, Some(call_id), transparency);
    }

    let call_site_ctxt = db.lookup_intern_macro_call(call_id).call_site.ctx;
    let mut call_site_ctxt = if transparency == Transparency::SemiTransparent {
        call_site_ctxt.normalize_to_macros_2_0(db)
    } else {
        call_site_ctxt.normalize_to_macro_rules(db)
    };

    if call_site_ctxt.is_root() {
        return apply_mark_internal(db, ctxt, Some(call_id), transparency);
    }

    // Otherwise, `expn_id` is a macros 1.0 definition and the call site is in a
    // macros 2.0 expansion, i.e., a macros 1.0 invocation is in a macros 2.0 definition.
    //
    // In this case, the tokens from the macros 1.0 definition inherit the hygiene
    // at their invocation. That is, we pretend that the macros 1.0 definition
    // was defined at its invocation (i.e., inside the macros 2.0 definition)
    // so that the macros 2.0 definition remains hygienic.
    //
    // See the example at `test/ui/hygiene/legacy_interaction.rs`.
    for (call_id, transparency) in ctxt.marks(db) {
        call_site_ctxt = apply_mark_internal(db, call_site_ctxt, call_id, transparency);
    }
    apply_mark_internal(db, call_site_ctxt, Some(call_id), transparency)
}

fn apply_mark_internal(
    db: &dyn ExpandDatabase,
    ctxt: SyntaxContextId,
    call_id: Option<MacroCallId>,
    transparency: Transparency,
) -> SyntaxContextId {
    let syntax_context_data = db.lookup_intern_syntax_context(ctxt);
    let mut opaque = syntax_context_data.opaque;
    let mut opaque_and_semitransparent = syntax_context_data.opaque_and_semitransparent;

    if transparency >= Transparency::Opaque {
        let parent = opaque;
        opaque = salsa::plumbing::get_query_table::<InternSyntaxContextQuery>(db).get_or_insert(
            (parent, call_id, transparency),
            |new_opaque| SyntaxContextData {
                outer_expn: call_id,
                outer_transparency: transparency,
                parent,
                opaque: new_opaque,
                opaque_and_semitransparent: new_opaque,
            },
        );
    }

    if transparency >= Transparency::SemiTransparent {
        let parent = opaque_and_semitransparent;
        opaque_and_semitransparent =
            salsa::plumbing::get_query_table::<InternSyntaxContextQuery>(db).get_or_insert(
                (parent, call_id, transparency),
                |new_opaque_and_semitransparent| SyntaxContextData {
                    outer_expn: call_id,
                    outer_transparency: transparency,
                    parent,
                    opaque,
                    opaque_and_semitransparent: new_opaque_and_semitransparent,
                },
            );
    }

    let parent = ctxt;
    db.intern_syntax_context(SyntaxContextData {
        outer_expn: call_id,
        outer_transparency: transparency,
        parent,
        opaque,
        opaque_and_semitransparent,
    })
}
pub trait SyntaxContextExt {
    fn normalize_to_macro_rules(self, db: &dyn ExpandDatabase) -> Self;
    fn normalize_to_macros_2_0(self, db: &dyn ExpandDatabase) -> Self;
    fn parent_ctxt(self, db: &dyn ExpandDatabase) -> Self;
    fn remove_mark(&mut self, db: &dyn ExpandDatabase) -> (Option<MacroCallId>, Transparency);
    fn outer_mark(self, db: &dyn ExpandDatabase) -> (Option<MacroCallId>, Transparency);
    fn marks(self, db: &dyn ExpandDatabase) -> Vec<(Option<MacroCallId>, Transparency)>;
}

impl SyntaxContextExt for SyntaxContextId {
    fn normalize_to_macro_rules(self, db: &dyn ExpandDatabase) -> Self {
        db.lookup_intern_syntax_context(self).opaque_and_semitransparent
    }
    fn normalize_to_macros_2_0(self, db: &dyn ExpandDatabase) -> Self {
        db.lookup_intern_syntax_context(self).opaque
    }
    fn parent_ctxt(self, db: &dyn ExpandDatabase) -> Self {
        db.lookup_intern_syntax_context(self).parent
    }
    fn outer_mark(self, db: &dyn ExpandDatabase) -> (Option<MacroCallId>, Transparency) {
        let data = db.lookup_intern_syntax_context(self);
        (data.outer_expn, data.outer_transparency)
    }
    fn remove_mark(&mut self, db: &dyn ExpandDatabase) -> (Option<MacroCallId>, Transparency) {
        let data = db.lookup_intern_syntax_context(*self);
        *self = data.parent;
        (data.outer_expn, data.outer_transparency)
    }
    fn marks(self, db: &dyn ExpandDatabase) -> Vec<(Option<MacroCallId>, Transparency)> {
        let mut marks = marks_rev(self, db).collect::<Vec<_>>();
        marks.reverse();
        marks
    }
}

// FIXME: Make this a SyntaxContextExt method once we have RPIT
pub fn marks_rev(
    ctxt: SyntaxContextId,
    db: &dyn ExpandDatabase,
) -> impl Iterator<Item = (Option<MacroCallId>, Transparency)> + '_ {
    iter::successors(Some(ctxt), move |&mark| {
        Some(mark.parent_ctxt(db)).filter(|&it| it != SyntaxContextId::ROOT)
    })
    .map(|ctx| ctx.outer_mark(db))
}

pub(crate) fn dump_syntax_contexts(db: &dyn ExpandDatabase) -> String {
    use crate::db::{InternMacroCallLookupQuery, InternSyntaxContextLookupQuery};
    use base_db::salsa::debug::DebugQueryTable;

    let mut s = String::from("Expansions:");
    let mut entries = InternMacroCallLookupQuery.in_db(db).entries::<Vec<_>>();
    entries.sort_by_key(|e| e.key);
    for e in entries {
        let id = e.key;
        let expn_data = e.value.as_ref().unwrap();
        s.push_str(&format!(
            "\n{:?}: parent: {:?}, call_site_ctxt: {:?}, def_site_ctxt: {:?}, kind: {:?}",
            id,
            expn_data.kind.file_id(),
            expn_data.call_site,
            SyntaxContextId::ROOT, // FIXME expn_data.def_site,
            expn_data.kind.descr(),
        ));
    }

    s.push_str("\n\nSyntaxContexts:\n");
    let mut entries = InternSyntaxContextLookupQuery.in_db(db).entries::<Vec<_>>();
    entries.sort_by_key(|e| e.key);
    for e in entries {
        struct SyntaxContextDebug<'a>(
            &'a dyn ExpandDatabase,
            SyntaxContextId,
            &'a SyntaxContextData,
        );

        impl<'a> std::fmt::Debug for SyntaxContextDebug<'a> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.2.fancy_debug(self.1, self.0, f)
            }
        }
        stdx::format_to!(s, "{:?}\n", SyntaxContextDebug(db, e.key, &e.value.unwrap()));
    }
    s
}
