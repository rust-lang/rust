//! Machinery for hygienic macros.
//!
//! Inspired by Matthew Flatt et al., “Macros That Work Together: Compile-Time Bindings, Partial
//! Expansion, and Definition Contexts,” *Journal of Functional Programming* 22, no. 2
//! (March 1, 2012): 181–216, <https://doi.org/10.1017/S0956796812000093>.
//!
//! Also see <https://rustc-dev-guide.rust-lang.org/macro-expansion.html#hygiene-and-hierarchies>
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

use std::{convert::identity, iter};

use span::{Edition, MacroCallId, Span, SyntaxContextId};

use crate::db::{ExpandDatabase, MacroCallWrapper};

pub use span::Transparency;

pub fn span_with_def_site_ctxt(
    db: &dyn ExpandDatabase,
    span: Span,
    expn_id: MacroCallId,
    edition: Edition,
) -> Span {
    span_with_ctxt_from_mark(db, span, expn_id, Transparency::Opaque, edition)
}

pub fn span_with_call_site_ctxt(
    db: &dyn ExpandDatabase,
    span: Span,
    expn_id: MacroCallId,
    edition: Edition,
) -> Span {
    span_with_ctxt_from_mark(db, span, expn_id, Transparency::Transparent, edition)
}

pub fn span_with_mixed_site_ctxt(
    db: &dyn ExpandDatabase,
    span: Span,
    expn_id: MacroCallId,
    edition: Edition,
) -> Span {
    span_with_ctxt_from_mark(db, span, expn_id, Transparency::SemiTransparent, edition)
}

fn span_with_ctxt_from_mark(
    db: &dyn ExpandDatabase,
    span: Span,
    expn_id: MacroCallId,
    transparency: Transparency,
    edition: Edition,
) -> Span {
    Span {
        ctx: apply_mark(db, SyntaxContextId::root(edition), expn_id, transparency, edition),
        ..span
    }
}

pub(super) fn apply_mark(
    db: &dyn ExpandDatabase,
    ctxt: span::SyntaxContextId,
    call_id: span::MacroCallId,
    transparency: Transparency,
    edition: Edition,
) -> SyntaxContextId {
    if transparency == Transparency::Opaque {
        return apply_mark_internal(db, ctxt, call_id, transparency, edition);
    }

    let call_site_ctxt = db.lookup_intern_macro_call(call_id).ctxt;
    let mut call_site_ctxt = if transparency == Transparency::SemiTransparent {
        call_site_ctxt.normalize_to_macros_2_0(db)
    } else {
        call_site_ctxt.normalize_to_macro_rules(db)
    };

    if call_site_ctxt.is_root() {
        return apply_mark_internal(db, ctxt, call_id, transparency, edition);
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
        call_site_ctxt = apply_mark_internal(db, call_site_ctxt, call_id, transparency, edition);
    }
    apply_mark_internal(db, call_site_ctxt, call_id, transparency, edition)
}

fn apply_mark_internal(
    db: &dyn ExpandDatabase,
    ctxt: SyntaxContextId,
    call_id: MacroCallId,
    transparency: Transparency,
    edition: Edition,
) -> SyntaxContextId {
    let call_id = Some(call_id);

    let mut opaque = ctxt.opaque(db);
    let mut opaque_and_semitransparent = ctxt.opaque_and_semitransparent(db);

    if transparency >= Transparency::Opaque {
        let parent = opaque;
        opaque =
            SyntaxContextId::new(db, call_id, transparency, edition, parent, identity, identity);
    }

    if transparency >= Transparency::SemiTransparent {
        let parent = opaque_and_semitransparent;
        opaque_and_semitransparent =
            SyntaxContextId::new(db, call_id, transparency, edition, parent, |_| opaque, identity);
    }

    let parent = ctxt;
    SyntaxContextId::new(
        db,
        call_id,
        transparency,
        edition,
        parent,
        |_| opaque,
        |_| opaque_and_semitransparent,
    )
}

pub trait SyntaxContextExt {
    fn normalize_to_macro_rules(self, db: &dyn ExpandDatabase) -> span::SyntaxContextId;
    fn normalize_to_macros_2_0(self, db: &dyn ExpandDatabase) -> span::SyntaxContextId;
    fn parent_ctxt(self, db: &dyn ExpandDatabase) -> span::SyntaxContextId;
    fn remove_mark(&mut self, db: &dyn ExpandDatabase)
        -> (Option<span::MacroCallId>, Transparency);
    fn outer_mark(self, db: &dyn ExpandDatabase) -> (Option<span::MacroCallId>, Transparency);
    fn marks(self, db: &dyn ExpandDatabase) -> Vec<(span::MacroCallId, Transparency)>;
    fn is_opaque(self, db: &dyn ExpandDatabase) -> bool;
}

impl SyntaxContextExt for SyntaxContextId {
    fn normalize_to_macro_rules(self, db: &dyn ExpandDatabase) -> span::SyntaxContextId {
        self.opaque_and_semitransparent(db)
    }
    fn normalize_to_macros_2_0(self, db: &dyn ExpandDatabase) -> span::SyntaxContextId {
        self.opaque(db)
    }
    fn parent_ctxt(self, db: &dyn ExpandDatabase) -> span::SyntaxContextId {
        self.parent(db)
    }
    fn outer_mark(self, db: &dyn ExpandDatabase) -> (Option<span::MacroCallId>, Transparency) {
        let data = self;
        (data.outer_expn(db), data.outer_transparency(db))
    }
    fn remove_mark(
        &mut self,
        db: &dyn ExpandDatabase,
    ) -> (Option<span::MacroCallId>, Transparency) {
        let data = *self;
        *self = data.parent(db);
        (data.outer_expn(db), data.outer_transparency(db))
    }
    fn marks(self, db: &dyn ExpandDatabase) -> Vec<(span::MacroCallId, Transparency)> {
        let mut marks = marks_rev(self, db).collect::<Vec<_>>();
        marks.reverse();
        marks
    }
    fn is_opaque(self, db: &dyn ExpandDatabase) -> bool {
        !self.is_root() && self.outer_transparency(db).is_opaque()
    }
}

// FIXME: Make this a SyntaxContextExt method once we have RPIT
pub fn marks_rev(
    ctxt: SyntaxContextId,
    db: &dyn ExpandDatabase,
) -> impl Iterator<Item = (span::MacroCallId, Transparency)> + '_ {
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
    let mut s = String::from("Expansions:");
    let entries =
        MacroCallWrapper::ingredient(db).entries(db.as_dyn_database()).collect::<Vec<_>>();
    for loc in entries {
        let expn_data = &loc.fields().0;

        s.push_str(&format!(
            "parent: {:?}, call_site_ctxt: {:?}, kind: {:?}",
            expn_data.kind.file_id(),
            expn_data.ctxt,
            expn_data.kind.descr(),
        ));
    }

    s.push_str("\n\nSyntaxContexts:\n");
    let entries = SyntaxContextId::ingredient(db).entries(db.as_dyn_database()).collect::<Vec<_>>();
    for e in entries {
        struct SyntaxContextDebug<'a>(
            &'a dyn ExpandDatabase,
            &'a span::SyntaxContextUnderlyingData,
        );

        impl std::fmt::Debug for SyntaxContextDebug<'_> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                fancy_debug(self.1, self.0, f)
            }
        }

        fn fancy_debug(
            this: &span::SyntaxContextUnderlyingData,
            db: &dyn ExpandDatabase,
            f: &mut std::fmt::Formatter<'_>,
        ) -> std::fmt::Result {
            write!(f, "parent: #{}, outer_mark: (", this.parent)?;
            match this.outer_expn {
                Some(id) => {
                    write!(f, "{:?}::{{{{expn{:?}}}}}", db.lookup_intern_macro_call(id).krate, id)?
                }
                None => write!(f, "root")?,
            }
            write!(f, ", {:?})", this.outer_transparency)
        }

        let dbg = SyntaxContextDebug(db, e.fields());
        stdx::format_to!(s, "{:?}\n", dbg);
    }
    s
}
