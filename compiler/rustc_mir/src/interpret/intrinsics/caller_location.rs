use std::convert::TryFrom;

use rustc_hir::lang_items::LangItem;
use rustc_middle::mir::TerminatorKind;
use rustc_middle::ty::subst::Subst;
use rustc_span::{Span, Symbol};
use rustc_target::abi::LayoutOf;

use crate::interpret::{
    intrinsics::{InterpCx, Machine},
    MPlaceTy, MemoryKind, Scalar,
};

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    /// Walks up the callstack from the intrinsic's callsite, searching for the first callsite in a
    /// frame which is not `#[track_caller]`.
    crate fn find_closest_untracked_caller_location(&self) -> Span {
        for frame in self.stack().iter().rev() {
            debug!("find_closest_untracked_caller_location: checking frame {:?}", frame.instance);

            // Assert that the frame we look at is actually executing code currently
            // (`loc` is `Err` when we are unwinding and the frame does not require cleanup).
            let loc = frame.loc.unwrap();

            // This could be a non-`Call` terminator (such as `Drop`), or not a terminator at all
            // (such as `box`). Use the normal span by default.
            let mut source_info = *frame.body.source_info(loc);

            // If this is a `Call` terminator, use the `fn_span` instead.
            let block = &frame.body.basic_blocks()[loc.block];
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

        bug!("no non-`#[track_caller]` frame found")
    }

    /// Allocate a `const core::panic::Location` with the provided filename and line/column numbers.
    crate fn alloc_caller_location(
        &mut self,
        filename: Symbol,
        line: u32,
        col: u32,
    ) -> MPlaceTy<'tcx, M::PointerTag> {
        let file = self.allocate_str(&filename.as_str(), MemoryKind::CallerLocation);
        let line = Scalar::from_u32(line);
        let col = Scalar::from_u32(col);

        // Allocate memory for `CallerLocation` struct.
        let loc_ty = self
            .tcx
            .type_of(self.tcx.require_lang_item(LangItem::PanicLocation, None))
            .subst(*self.tcx, self.tcx.mk_substs([self.tcx.lifetimes.re_erased.into()].iter()));
        let loc_layout = self.layout_of(loc_ty).unwrap();
        let location = self.allocate(loc_layout, MemoryKind::CallerLocation);

        // Initialize fields.
        self.write_immediate(file.to_ref(), self.mplace_field(location, 0).unwrap().into())
            .expect("writing to memory we just allocated cannot fail");
        self.write_scalar(line, self.mplace_field(location, 1).unwrap().into())
            .expect("writing to memory we just allocated cannot fail");
        self.write_scalar(col, self.mplace_field(location, 2).unwrap().into())
            .expect("writing to memory we just allocated cannot fail");

        location
    }

    crate fn location_triple_for_span(&self, span: Span) -> (Symbol, u32, u32) {
        let topmost = span.ctxt().outer_expn().expansion_cause().unwrap_or(span);
        let caller = self.tcx.sess.source_map().lookup_char_pos(topmost.lo());
        (
            Symbol::intern(&caller.file.name.to_string()),
            u32::try_from(caller.line).unwrap(),
            u32::try_from(caller.col_display).unwrap().checked_add(1).unwrap(),
        )
    }

    pub fn alloc_caller_location_for_span(&mut self, span: Span) -> MPlaceTy<'tcx, M::PointerTag> {
        let (file, line, column) = self.location_triple_for_span(span);
        self.alloc_caller_location(file, line, column)
    }
}
