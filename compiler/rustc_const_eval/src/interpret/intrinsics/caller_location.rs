use rustc_ast::Mutability;
use rustc_hir::lang_items::LangItem;
use rustc_middle::mir::TerminatorKind;
use rustc_middle::ty::layout::LayoutOf;
use rustc_span::{Span, Symbol};

use crate::interpret::{
    intrinsics::{InterpCx, Machine},
    MPlaceTy, MemoryKind, Scalar,
};

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    /// Walks up the callstack from the intrinsic's callsite, searching for the first callsite in a
    /// frame which is not `#[track_caller]`.
    pub(crate) fn find_closest_untracked_caller_location(&self) -> Span {
        for frame in self.stack().iter().rev() {
            debug!("find_closest_untracked_caller_location: checking frame {:?}", frame.instance);

            // Assert that the frame we look at is actually executing code currently
            // (`loc` is `Right` when we are unwinding and the frame does not require cleanup).
            let loc = frame.loc.left().unwrap();

            // This could be a non-`Call` terminator (such as `Drop`), or not a terminator at all
            // (such as `box`). Use the normal span by default.
            let mut source_info = *frame.body.source_info(loc);

            // If this is a `Call` terminator, use the `fn_span` instead.
            let block = &frame.body.basic_blocks[loc.block];
            if loc.statement_index == block.statements.len() {
                debug!(
                    "find_closest_untracked_caller_location: got terminator {:?} ({:?})",
                    block.terminator(),
                    block.terminator().kind
                );
                if let TerminatorKind::Call { fn_span, .. } = block.terminator().kind {
                    source_info.span = fn_span;
                }
            }

            // Walk up the `SourceScope`s, in case some of them are from MIR inlining.
            // If so, the starting `source_info.span` is in the innermost inlined
            // function, and will be replaced with outer callsite spans as long
            // as the inlined functions were `#[track_caller]`.
            loop {
                let scope_data = &frame.body.source_scopes[source_info.scope];

                if let Some((callee, callsite_span)) = scope_data.inlined {
                    // Stop inside the most nested non-`#[track_caller]` function,
                    // before ever reaching its caller (which is irrelevant).
                    if !callee.def.requires_caller_location(*self.tcx) {
                        return source_info.span;
                    }
                    source_info.span = callsite_span;
                }

                // Skip past all of the parents with `inlined: None`.
                match scope_data.inlined_parent_scope {
                    Some(parent) => source_info.scope = parent,
                    None => break,
                }
            }

            // Stop inside the most nested non-`#[track_caller]` function,
            // before ever reaching its caller (which is irrelevant).
            if !frame.instance.def.requires_caller_location(*self.tcx) {
                return source_info.span;
            }
        }

        span_bug!(self.cur_span(), "no non-`#[track_caller]` frame found")
    }

    /// Allocate a `const core::panic::Location` with the provided filename and line/column numbers.
    pub(crate) fn alloc_caller_location(
        &mut self,
        filename: Symbol,
        line: u32,
        col: u32,
    ) -> MPlaceTy<'tcx, M::Provenance> {
        let loc_details = &self.tcx.sess.opts.unstable_opts.location_detail;
        // This can fail if rustc runs out of memory right here. Trying to emit an error would be
        // pointless, since that would require allocating more memory than these short strings.
        let file = if loc_details.file {
            self.allocate_str(filename.as_str(), MemoryKind::CallerLocation, Mutability::Not)
                .unwrap()
        } else {
            // FIXME: This creates a new allocation each time. It might be preferable to
            // perform this allocation only once, and re-use the `MPlaceTy`.
            // See https://github.com/rust-lang/rust/pull/89920#discussion_r730012398
            self.allocate_str("<redacted>", MemoryKind::CallerLocation, Mutability::Not).unwrap()
        };
        let line = if loc_details.line { Scalar::from_u32(line) } else { Scalar::from_u32(0) };
        let col = if loc_details.column { Scalar::from_u32(col) } else { Scalar::from_u32(0) };

        // Allocate memory for `CallerLocation` struct.
        let loc_ty = self
            .tcx
            .type_of(self.tcx.require_lang_item(LangItem::PanicLocation, None))
            .subst(*self.tcx, self.tcx.intern_substs(&[self.tcx.lifetimes.re_erased.into()]));
        let loc_layout = self.layout_of(loc_ty).unwrap();
        let location = self.allocate(loc_layout, MemoryKind::CallerLocation).unwrap();

        // Initialize fields.
        self.write_immediate(file.to_ref(self), &self.mplace_field(&location, 0).unwrap().into())
            .expect("writing to memory we just allocated cannot fail");
        self.write_scalar(line, &self.mplace_field(&location, 1).unwrap().into())
            .expect("writing to memory we just allocated cannot fail");
        self.write_scalar(col, &self.mplace_field(&location, 2).unwrap().into())
            .expect("writing to memory we just allocated cannot fail");

        location
    }

    pub(crate) fn location_triple_for_span(&self, span: Span) -> (Symbol, u32, u32) {
        let topmost = span.ctxt().outer_expn().expansion_cause().unwrap_or(span);
        let caller = self.tcx.sess.source_map().lookup_char_pos(topmost.lo());
        (
            Symbol::intern(&caller.file.name.prefer_remapped().to_string_lossy()),
            u32::try_from(caller.line).unwrap(),
            u32::try_from(caller.col_display).unwrap().checked_add(1).unwrap(),
        )
    }

    pub fn alloc_caller_location_for_span(&mut self, span: Span) -> MPlaceTy<'tcx, M::Provenance> {
        let (file, line, column) = self.location_triple_for_span(span);
        self.alloc_caller_location(file, line, column)
    }
}
