use rustc::middle::lang_items::PanicLocationLangItem;
use rustc::ty::subst::Subst;
use rustc_target::abi::LayoutOf;
use syntax_pos::{Symbol, Span};

use crate::interpret::{Scalar, MemoryKind, MPlaceTy, intrinsics::{InterpCx, Machine}};

impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
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
        let loc_ty = self.tcx.type_of(self.tcx.require_lang_item(PanicLocationLangItem, None))
            .subst(*self.tcx, self.tcx.mk_substs([self.tcx.lifetimes.re_static.into()].iter()));
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

    pub fn alloc_caller_location_for_span(
        &mut self,
        span: Span,
    ) -> MPlaceTy<'tcx, M::PointerTag> {
        let topmost = span.ctxt().outer_expn().expansion_cause().unwrap_or(span);
        let caller = self.tcx.sess.source_map().lookup_char_pos(topmost.lo());
        self.alloc_caller_location(
            Symbol::intern(&caller.file.name.to_string()),
            caller.line as u32,
            caller.col_display as u32 + 1,
        )
    }
}
