//! # The rustc_public's IR Visitor
//!
//! ## Overview
//!
//! We currently only support an immutable visitor.
//! The structure of this visitor is similar to the ones internal to `rustc`,
//! and it follows the following conventions:
//!
//! For every mir item, the trait has a `visit_<item>` and a `super_<item>` method.
//! - `visit_<item>`, by default, calls `super_<item>`
//! - `super_<item>`, by default, destructures the `<item>` and calls `visit_<sub_item>` for
//!   all sub-items that compose the original item.
//!
//! In order to implement a visitor, override the `visit_*` methods for the types you are
//! interested in analyzing, and invoke (within that method call)
//! `self.super_*` to continue to the traverse.
//! Avoid calling `super` methods in other circumstances.
//!
//! For the most part, we do not destructure things external to the
//! MIR, e.g., types, spans, etc, but simply visit them and stop.
//! This avoids duplication with other visitors like `TypeFoldable`.
//!
//! ## Updating
//!
//! The code is written in a very deliberate style intended to minimize
//! the chance of things being overlooked.
//!
//! Use pattern matching to reference fields and ensure that all
//! matches are exhaustive.
//!
//! For this to work, ALL MATCHES MUST BE EXHAUSTIVE IN FIELDS AND VARIANTS.
//! That means you never write `..` to skip over fields, nor do you write `_`
//! to skip over variants in a `match`.
//!
//! The only place that `_` is acceptable is to match a field (or
//! variant argument) that does not require visiting.

use crate::mir::*;
use crate::ty::{GenericArgs, MirConst, Region, Ty, TyConst};
use crate::{Error, Opaque, Span};

macro_rules! make_mir_visitor {
    ($visitor_trait_name:ident, $($mutability:ident)?) => {
        pub trait $visitor_trait_name {
            fn visit_body(&mut self, body: &$($mutability)? Body) {
                self.super_body(body)
            }

            fn visit_basic_block(&mut self, bb: &$($mutability)? BasicBlock) {
                self.super_basic_block(bb)
            }

            fn visit_ret_decl(&mut self, local: Local, decl: &$($mutability)? LocalDecl) {
                self.super_ret_decl(local, decl)
            }

            fn visit_arg_decl(&mut self, local: Local, decl: &$($mutability)? LocalDecl) {
                self.super_arg_decl(local, decl)
            }

            fn visit_local_decl(&mut self, local: Local, decl: &$($mutability)? LocalDecl) {
                self.super_local_decl(local, decl)
            }

            fn visit_statement(&mut self, stmt: &$($mutability)? Statement, location: Location) {
                self.super_statement(stmt, location)
            }

            fn visit_terminator(&mut self, term: &$($mutability)? Terminator, location: Location) {
                self.super_terminator(term, location)
            }

            fn visit_span(&mut self, span: &$($mutability)? Span) {
                self.super_span(span)
            }

            fn visit_place(&mut self, place: &$($mutability)? Place, ptx: PlaceContext, location: Location) {
                self.super_place(place, ptx, location)
            }

            visit_place_fns!($($mutability)?);

            fn visit_local(&mut self, local: &$($mutability)? Local, ptx: PlaceContext, location: Location) {
                let _ = (local, ptx, location);
            }

            fn visit_rvalue(&mut self, rvalue: &$($mutability)? Rvalue, location: Location) {
                self.super_rvalue(rvalue, location)
            }

            fn visit_operand(&mut self, operand: &$($mutability)? Operand, location: Location) {
                self.super_operand(operand, location)
            }

            fn visit_user_type_projection(&mut self, projection: &$($mutability)? UserTypeProjection) {
                self.super_user_type_projection(projection)
            }

            fn visit_ty(&mut self, ty: &$($mutability)? Ty, location: Location) {
                let _ = location;
                self.super_ty(ty)
            }

            fn visit_const_operand(&mut self, constant: &$($mutability)? ConstOperand, location: Location) {
                self.super_const_operand(constant, location)
            }

            fn visit_mir_const(&mut self, constant: &$($mutability)? MirConst, location: Location) {
                self.super_mir_const(constant, location)
            }

            fn visit_ty_const(&mut self, constant: &$($mutability)? TyConst, location: Location) {
                let _ = location;
                self.super_ty_const(constant)
            }

            fn visit_region(&mut self, region: &$($mutability)? Region, location: Location) {
                let _ = location;
                self.super_region(region)
            }

            fn visit_args(&mut self, args: &$($mutability)? GenericArgs, location: Location) {
                let _ = location;
                self.super_args(args)
            }

            fn visit_assert_msg(&mut self, msg: &$($mutability)? AssertMessage, location: Location) {
                self.super_assert_msg(msg, location)
            }

            fn visit_var_debug_info(&mut self, var_debug_info: &$($mutability)? VarDebugInfo) {
                self.super_var_debug_info(var_debug_info);
            }

            fn super_body(&mut self, body: &$($mutability)? Body) {
                super_body!(self, body, $($mutability)?);
            }

            fn super_basic_block(&mut self, bb: &$($mutability)? BasicBlock) {
                let BasicBlock { statements, terminator } = bb;
                for stmt in statements {
                    self.visit_statement(stmt, Location(stmt.span));
                }
                self.visit_terminator(terminator, Location(terminator.span));
            }

            fn super_local_decl(&mut self, local: Local, decl: &$($mutability)? LocalDecl) {
                let _ = local;
                let LocalDecl { ty, span, .. } = decl;
                self.visit_ty(ty, Location(*span));
            }

            fn super_ret_decl(&mut self, local: Local, decl: &$($mutability)? LocalDecl) {
                self.super_local_decl(local, decl)
            }

            fn super_arg_decl(&mut self, local: Local, decl: &$($mutability)? LocalDecl) {
                self.super_local_decl(local, decl)
            }

            fn super_statement(&mut self, stmt: &$($mutability)? Statement, location: Location) {
                let Statement { kind, span } = stmt;
                self.visit_span(span);
                match kind {
                    StatementKind::Assign(place, rvalue) => {
                        self.visit_place(place, PlaceContext::MUTATING, location);
                        self.visit_rvalue(rvalue, location);
                    }
                    StatementKind::FakeRead(_, place) | StatementKind::PlaceMention(place) => {
                        self.visit_place(place, PlaceContext::NON_MUTATING, location);
                    }
                    StatementKind::SetDiscriminant { place, .. }
                    | StatementKind::Deinit(place)
                    | StatementKind::Retag(_, place) => {
                        self.visit_place(place, PlaceContext::MUTATING, location);
                    }
                    StatementKind::StorageLive(local) | StatementKind::StorageDead(local) => {
                        self.visit_local(local, PlaceContext::NON_USE, location);
                    }
                    StatementKind::AscribeUserType { place, projections, variance: _ } => {
                        self.visit_place(place, PlaceContext::NON_USE, location);
                        self.visit_user_type_projection(projections);
                    }
                    StatementKind::Coverage(coverage) => visit_opaque(coverage),
                    StatementKind::Intrinsic(intrisic) => match intrisic {
                        NonDivergingIntrinsic::Assume(operand) => {
                            self.visit_operand(operand, location);
                        }
                        NonDivergingIntrinsic::CopyNonOverlapping(CopyNonOverlapping {
                            src,
                            dst,
                            count,
                        }) => {
                            self.visit_operand(src, location);
                            self.visit_operand(dst, location);
                            self.visit_operand(count, location);
                        }
                    },
                    StatementKind::ConstEvalCounter | StatementKind::Nop => {}
                }
            }

            fn super_terminator(&mut self, term: &$($mutability)? Terminator, location: Location) {
                let Terminator { kind, span } = term;
                self.visit_span(span);
                match kind {
                    TerminatorKind::Goto { .. }
                    | TerminatorKind::Resume
                    | TerminatorKind::Abort
                    | TerminatorKind::Unreachable => {}
                    TerminatorKind::Assert { cond, expected: _, msg, target: _, unwind: _ } => {
                        self.visit_operand(cond, location);
                        self.visit_assert_msg(msg, location);
                    }
                    TerminatorKind::Drop { place, target: _, unwind: _ } => {
                        self.visit_place(place, PlaceContext::MUTATING, location);
                    }
                    TerminatorKind::Call { func, args, destination, target: _, unwind: _ } => {
                        self.visit_operand(func, location);
                        for arg in args {
                            self.visit_operand(arg, location);
                        }
                        self.visit_place(destination, PlaceContext::MUTATING, location);
                    }
                    TerminatorKind::InlineAsm { operands, .. } => {
                        for op in operands {
                            let InlineAsmOperand { in_value, out_place, raw_rpr: _ } = op;
                            if let Some(input) = in_value {
                                self.visit_operand(input, location);
                            }
                            if let Some(output) = out_place {
                                self.visit_place(output, PlaceContext::MUTATING, location);
                            }
                        }
                    }
                    TerminatorKind::Return => {
                        let $($mutability)? local = RETURN_LOCAL;
                        self.visit_local(&$($mutability)? local, PlaceContext::NON_MUTATING, location);
                    }
                    TerminatorKind::SwitchInt { discr, targets: _ } => {
                        self.visit_operand(discr, location);
                    }
                }
            }

            fn super_span(&mut self, span: &$($mutability)? Span) {
                let _ = span;
            }

            fn super_rvalue(&mut self, rvalue: &$($mutability)? Rvalue, location: Location) {
                match rvalue {
                    Rvalue::AddressOf(mutability, place) => {
                        let pcx = PlaceContext { is_mut: *mutability == RawPtrKind::Mut };
                        self.visit_place(place, pcx, location);
                    }
                    Rvalue::Aggregate(_, operands) => {
                        for op in operands {
                            self.visit_operand(op, location);
                        }
                    }
                    Rvalue::BinaryOp(_, lhs, rhs) | Rvalue::CheckedBinaryOp(_, lhs, rhs) => {
                        self.visit_operand(lhs, location);
                        self.visit_operand(rhs, location);
                    }
                    Rvalue::Cast(_, op, ty) => {
                        self.visit_operand(op, location);
                        self.visit_ty(ty, location);
                    }
                    Rvalue::CopyForDeref(place) | Rvalue::Discriminant(place) | Rvalue::Len(place) => {
                        self.visit_place(place, PlaceContext::NON_MUTATING, location);
                    }
                    Rvalue::Ref(region, kind, place) => {
                        self.visit_region(region, location);
                        let pcx = PlaceContext { is_mut: matches!(kind, BorrowKind::Mut { .. }) };
                        self.visit_place(place, pcx, location);
                    }
                    Rvalue::Repeat(op, constant) => {
                        self.visit_operand(op, location);
                        self.visit_ty_const(constant, location);
                    }
                    Rvalue::ShallowInitBox(op, ty) => {
                        self.visit_ty(ty, location);
                        self.visit_operand(op, location)
                    }
                    Rvalue::ThreadLocalRef(_) => {}
                    Rvalue::NullaryOp(_, ty) => {
                        self.visit_ty(ty, location);
                    }
                    Rvalue::UnaryOp(_, op) | Rvalue::Use(op) => {
                        self.visit_operand(op, location);
                    }
                }
            }

            fn super_operand(&mut self, operand: &$($mutability)? Operand, location: Location) {
                match operand {
                    Operand::Copy(place) | Operand::Move(place) => {
                        self.visit_place(place, PlaceContext::NON_MUTATING, location)
                    }
                    Operand::Constant(constant) => {
                        self.visit_const_operand(constant, location);
                    }
                }
            }

            fn super_user_type_projection(&mut self, projection: &$($mutability)? UserTypeProjection) {
                // This is a no-op on mir::Visitor.
                let _ = projection;
            }

            fn super_ty(&mut self, ty: &$($mutability)? Ty) {
                let _ = ty;
            }

            fn super_const_operand(&mut self, constant: &$($mutability)? ConstOperand, location: Location) {
                let ConstOperand { span, user_ty: _, const_ } = constant;
                self.visit_span(span);
                self.visit_mir_const(const_, location);
            }

            fn super_mir_const(&mut self, constant: &$($mutability)? MirConst, location: Location) {
                let MirConst { kind: _, ty, id: _ } = constant;
                self.visit_ty(ty, location);
            }

            fn super_ty_const(&mut self, constant: &$($mutability)? TyConst) {
                let _ = constant;
            }

            fn super_region(&mut self, region: &$($mutability)? Region) {
                let _ = region;
            }

            fn super_args(&mut self, args: &$($mutability)? GenericArgs) {
                let _ = args;
            }

            fn super_var_debug_info(&mut self, var_debug_info: &$($mutability)? VarDebugInfo) {
                let VarDebugInfo { source_info, composite, value, name: _, argument_index: _ } =
                    var_debug_info;
                self.visit_span(&$($mutability)? source_info.span);
                let location = Location(source_info.span);
                if let Some(composite) = composite {
                    self.visit_ty(&$($mutability)? composite.ty, location);
                }
                match value {
                    VarDebugInfoContents::Place(place) => {
                        self.visit_place(place, PlaceContext::NON_USE, location);
                    }
                    VarDebugInfoContents::Const(constant) => {
                        self.visit_mir_const(&$($mutability)? constant.const_, location);
                    }
                }
            }

            fn super_assert_msg(&mut self, msg: &$($mutability)? AssertMessage, location: Location) {
                match msg {
                    AssertMessage::BoundsCheck { len, index } => {
                        self.visit_operand(len, location);
                        self.visit_operand(index, location);
                    }
                    AssertMessage::Overflow(_, left, right) => {
                        self.visit_operand(left, location);
                        self.visit_operand(right, location);
                    }
                    AssertMessage::OverflowNeg(op)
                    | AssertMessage::DivisionByZero(op)
                    | AssertMessage::RemainderByZero(op)
                    | AssertMessage::InvalidEnumConstruction(op) => {
                        self.visit_operand(op, location);
                    }
                    AssertMessage::ResumedAfterReturn(_)
                    | AssertMessage::ResumedAfterPanic(_)
                    | AssertMessage::NullPointerDereference
                    | AssertMessage::ResumedAfterDrop(_) => {
                        //nothing to visit
                    }
                    AssertMessage::MisalignedPointerDereference { required, found } => {
                        self.visit_operand(required, location);
                        self.visit_operand(found, location);
                    }
                }
            }
        }
    };
}

macro_rules! super_body {
    ($self:ident, $body:ident, mut) => {
        for bb in $body.blocks.iter_mut() {
            $self.visit_basic_block(bb);
        }

        $self.visit_ret_decl(RETURN_LOCAL, $body.ret_local_mut());

        for (idx, arg) in $body.arg_locals_mut().iter_mut().enumerate() {
            $self.visit_arg_decl(idx + 1, arg)
        }

        let local_start = $body.arg_count + 1;
        for (idx, arg) in $body.inner_locals_mut().iter_mut().enumerate() {
            $self.visit_local_decl(idx + local_start, arg)
        }

        for info in $body.var_debug_info.iter_mut() {
            $self.visit_var_debug_info(info);
        }

        $self.visit_span(&mut $body.span)
    };

    ($self:ident, $body:ident, ) => {
        let Body { blocks, locals: _, arg_count, var_debug_info, spread_arg: _, span } = $body;

        for bb in blocks {
            $self.visit_basic_block(bb);
        }

        $self.visit_ret_decl(RETURN_LOCAL, $body.ret_local());

        for (idx, arg) in $body.arg_locals().iter().enumerate() {
            $self.visit_arg_decl(idx + 1, arg)
        }

        let local_start = arg_count + 1;
        for (idx, arg) in $body.inner_locals().iter().enumerate() {
            $self.visit_local_decl(idx + local_start, arg)
        }

        for info in var_debug_info.iter() {
            $self.visit_var_debug_info(info);
        }

        $self.visit_span(span)
    };
}

macro_rules! visit_place_fns {
    (mut) => {
        fn super_place(&mut self, place: &mut Place, ptx: PlaceContext, location: Location) {
            self.visit_local(&mut place.local, ptx, location);

            for elem in place.projection.iter_mut() {
                self.visit_projection_elem(elem, ptx, location);
            }
        }

        // We don't have to replicate the `process_projection()` like we did in
        // `rustc_middle::mir::visit.rs` here because the `projection` field in `Place`
        // of Stable-MIR is not an immutable borrow, unlike in `Place` of MIR.
        fn visit_projection_elem(
            &mut self,
            elem: &mut ProjectionElem,
            ptx: PlaceContext,
            location: Location,
        ) {
            self.super_projection_elem(elem, ptx, location)
        }

        fn super_projection_elem(
            &mut self,
            elem: &mut ProjectionElem,
            ptx: PlaceContext,
            location: Location,
        ) {
            match elem {
                ProjectionElem::Deref => {}
                ProjectionElem::Field(_idx, ty) => self.visit_ty(ty, location),
                ProjectionElem::Index(local) => self.visit_local(local, ptx, location),
                ProjectionElem::ConstantIndex { offset: _, min_length: _, from_end: _ } => {}
                ProjectionElem::Subslice { from: _, to: _, from_end: _ } => {}
                ProjectionElem::Downcast(_idx) => {}
                ProjectionElem::OpaqueCast(ty) => self.visit_ty(ty, location),
            }
        }
    };

    () => {
        fn super_place(&mut self, place: &Place, ptx: PlaceContext, location: Location) {
            self.visit_local(&place.local, ptx, location);

            for (idx, elem) in place.projection.iter().enumerate() {
                let place_ref =
                    PlaceRef { local: place.local, projection: &place.projection[..idx] };
                self.visit_projection_elem(place_ref, elem, ptx, location);
            }
        }

        fn visit_projection_elem<'a>(
            &mut self,
            place_ref: PlaceRef<'a>,
            elem: &ProjectionElem,
            ptx: PlaceContext,
            location: Location,
        ) {
            let _ = place_ref;
            self.super_projection_elem(elem, ptx, location);
        }

        fn super_projection_elem(
            &mut self,
            elem: &ProjectionElem,
            ptx: PlaceContext,
            location: Location,
        ) {
            match elem {
                ProjectionElem::Deref => {}
                ProjectionElem::Field(_idx, ty) => self.visit_ty(ty, location),
                ProjectionElem::Index(local) => self.visit_local(local, ptx, location),
                ProjectionElem::ConstantIndex { offset: _, min_length: _, from_end: _ } => {}
                ProjectionElem::Subslice { from: _, to: _, from_end: _ } => {}
                ProjectionElem::Downcast(_idx) => {}
                ProjectionElem::OpaqueCast(ty) => self.visit_ty(ty, location),
            }
        }
    };
}

make_mir_visitor!(MirVisitor,);
make_mir_visitor!(MutMirVisitor, mut);

/// This function is a no-op that gets used to ensure this visitor is kept up-to-date.
///
/// The idea is that whenever we replace an Opaque type by a real type, the compiler will fail
/// when trying to invoke `visit_opaque`.
///
/// If you are here because your compilation is broken, replace the failing call to `visit_opaque()`
/// by a `visit_<CONSTRUCT>` for your construct.
fn visit_opaque(_: &Opaque) {}

/// The location of a statement / terminator in the code and the CFG.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Location(Span);

impl Location {
    pub fn span(&self) -> Span {
        self.0
    }
}

/// Location of the statement at the given index for a given basic block. Assumes that `stmt_idx`
/// and `bb_idx` are valid for a given body.
pub fn statement_location(body: &Body, bb_idx: &BasicBlockIdx, stmt_idx: usize) -> Location {
    let bb = &body.blocks[*bb_idx];
    let stmt = &bb.statements[stmt_idx];
    Location(stmt.span)
}

/// Location of the terminator for a given basic block. Assumes that `bb_idx` is valid for a given
/// body.
pub fn terminator_location(body: &Body, bb_idx: &BasicBlockIdx) -> Location {
    let bb = &body.blocks[*bb_idx];
    let terminator = &bb.terminator;
    Location(terminator.span)
}

/// Reference to a place used to represent a partial projection.
pub struct PlaceRef<'a> {
    pub local: Local,
    pub projection: &'a [ProjectionElem],
}

impl PlaceRef<'_> {
    /// Get the type of this place.
    pub fn ty(&self, locals: &[LocalDecl]) -> Result<Ty, Error> {
        self.projection.iter().try_fold(locals[self.local].ty, |place_ty, elem| elem.ty(place_ty))
    }
}

/// Information about a place's usage.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PlaceContext {
    /// Whether the access is mutable or not. Keep this private so we can increment the type in a
    /// backward compatible manner.
    is_mut: bool,
}

impl PlaceContext {
    const MUTATING: Self = PlaceContext { is_mut: true };
    const NON_MUTATING: Self = PlaceContext { is_mut: false };
    const NON_USE: Self = PlaceContext { is_mut: false };

    pub fn is_mutating(&self) -> bool {
        self.is_mut
    }
}
