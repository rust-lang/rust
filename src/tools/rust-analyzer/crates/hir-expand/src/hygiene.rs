//! Machinery for hygienic macros.
//!
//! Inspired by Matthew Flatt et al., “Macros That Work Together: Compile-Time Bindings, Partial
//! Expansion, and Definition Contexts,” *Journal of Functional Programming* 22, no. 2
//! (March 1, 2012): 181–216, <https://doi.org/10.1017/S0956796812000093>.
//!
//! Also see https://rustc-dev-guide.rust-lang.org/macro-expansion.html#hygiene-and-hierarchies
//!
//! # The Expansion Order Hierarchy
//!
//! `ExpnData` in rustc, rust-analyzer's version is [`MacroCallLoc`]. Traversing the hierarchy
//! upwards can be achieved by walking up [`MacroCallLoc::kind`]'s contained file id, as
//! [`MacroFile`]s are interned [`MacroCallLoc`]s.
//!
//! # The Macro Definition Hierarchy
//!
//! `SyntaxContextData` in rustc and rust-analyzer. Basically the same in both.
//!
//! # The Call-site Hierarchy
//!
//! `ExpnData::call_site` in rustc, [`MacroCallLoc::call_site`] in rust-analyzer.
// FIXME: Move this into the span crate? Not quite possible today as that depends on `MacroCallLoc`
// which contains a bunch of unrelated things

use std::iter;

use span::{MacroCallId, Span, SyntaxContextData, SyntaxContextId};

use crate::db::{ExpandDatabase, InternSyntaxContextQuery};

pub use span::Transparency;

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
        return apply_mark_internal(db, ctxt, call_id, transparency);
    }

    let call_site_ctxt = db.lookup_intern_macro_call(call_id).ctxt;
    let mut call_site_ctxt = if transparency == Transparency::SemiTransparent {
        call_site_ctxt.normalize_to_macros_2_0(db)
    } else {
        call_site_ctxt.normalize_to_macro_rules(db)
    };

    if call_site_ctxt.is_root() {
        return apply_mark_internal(db, ctxt, call_id, transparency);
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
    apply_mark_internal(db, call_site_ctxt, call_id, transparency)
}

fn apply_mark_internal(
    db: &dyn ExpandDatabase,
    ctxt: SyntaxContextId,
    call_id: MacroCallId,
    transparency: Transparency,
) -> SyntaxContextId {
    use base_db::salsa;

    let call_id = Some(call_id);

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
    fn marks(self, db: &dyn ExpandDatabase) -> Vec<(MacroCallId, Transparency)>;
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
    fn marks(self, db: &dyn ExpandDatabase) -> Vec<(MacroCallId, Transparency)> {
        let mut marks = marks_rev(self, db).collect::<Vec<_>>();
        marks.reverse();
        marks
    }
}

// FIXME: Make this a SyntaxContextExt method once we have RPIT
pub fn marks_rev(
    ctxt: SyntaxContextId,
    db: &dyn ExpandDatabase,
) -> impl Iterator<Item = (MacroCallId, Transparency)> + '_ {
    iter::successors(Some(ctxt), move |&mark| Some(mark.parent_ctxt(db)))
        .take_while(|&it| !it.is_root())
        .map(|ctx| {
            let mark = ctx.outer_mark(db);
            // We stop before taking the root expansion, as such we cannot encounter a `None` outer
            // expansion, as only the ROOT has it.
            (mark.0.unwrap(), mark.1)
        })
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
            "\n{:?}: parent: {:?}, call_site_ctxt: {:?}, kind: {:?}",
            id,
            expn_data.kind.file_id(),
            expn_data.ctxt,
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
                fancy_debug(self.2, self.1, self.0, f)
            }
        }

        fn fancy_debug(
            this: &SyntaxContextData,
            self_id: SyntaxContextId,
            db: &dyn ExpandDatabase,
            f: &mut std::fmt::Formatter<'_>,
        ) -> std::fmt::Result {
            write!(f, "#{self_id} parent: #{}, outer_mark: (", this.parent)?;
            match this.outer_expn {
                Some(id) => {
                    write!(f, "{:?}::{{{{expn{:?}}}}}", db.lookup_intern_macro_call(id).krate, id)?
                }
                None => write!(f, "root")?,
            }
            write!(f, ", {:?})", this.outer_transparency)
        }

        stdx::format_to!(s, "{:?}\n", SyntaxContextDebug(db, e.key, &e.value.unwrap()));
    }
    s
}
