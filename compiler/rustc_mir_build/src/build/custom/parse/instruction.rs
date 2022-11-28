use rustc_middle::{mir::*, thir::*, ty};

use super::{parse_by_kind, PResult, ParseCtxt};

impl<'tcx, 'body> ParseCtxt<'tcx, 'body> {
    pub fn parse_statement(&self, expr_id: ExprId) -> PResult<StatementKind<'tcx>> {
        parse_by_kind!(self, expr_id, _, "statement",
            @call("mir_retag", args) => {
                Ok(StatementKind::Retag(RetagKind::Default, Box::new(self.parse_place(args[0])?)))
            },
            @call("mir_retag_raw", args) => {
                Ok(StatementKind::Retag(RetagKind::Raw, Box::new(self.parse_place(args[0])?)))
            },
            ExprKind::Assign { lhs, rhs } => {
                let lhs = self.parse_place(*lhs)?;
                let rhs = self.parse_rvalue(*rhs)?;
                Ok(StatementKind::Assign(Box::new((lhs, rhs))))
            },
        )
    }

    pub fn parse_terminator(&self, expr_id: ExprId) -> PResult<TerminatorKind<'tcx>> {
        parse_by_kind!(self, expr_id, _, "terminator",
            @call("mir_return", _args) => {
                Ok(TerminatorKind::Return)
            },
            @call("mir_goto", args) => {
                Ok(TerminatorKind::Goto { target: self.parse_block(args[0])? } )
            },
        )
    }

    fn parse_rvalue(&self, expr_id: ExprId) -> PResult<Rvalue<'tcx>> {
        parse_by_kind!(self, expr_id, _, "rvalue",
            ExprKind::Borrow { borrow_kind, arg } => Ok(
                Rvalue::Ref(self.tcx.lifetimes.re_erased, *borrow_kind, self.parse_place(*arg)?)
            ),
            ExprKind::AddressOf { mutability, arg } => Ok(
                Rvalue::AddressOf(*mutability, self.parse_place(*arg)?)
            ),
            _ => self.parse_operand(expr_id).map(Rvalue::Use),
        )
    }

    fn parse_operand(&self, expr_id: ExprId) -> PResult<Operand<'tcx>> {
        parse_by_kind!(self, expr_id, expr, "operand",
            @call("mir_move", args) => self.parse_place(args[0]).map(Operand::Move),
            ExprKind::Literal { .. }
            | ExprKind::NamedConst { .. }
            | ExprKind::NonHirLiteral { .. }
            | ExprKind::ZstLiteral { .. }
            | ExprKind::ConstParam { .. }
            | ExprKind::ConstBlock { .. } => {
                Ok(Operand::Constant(Box::new(
                    crate::build::expr::as_constant::as_constant_inner(expr, |_| None, self.tcx)
                )))
            },
            _ => self.parse_place(expr_id).map(Operand::Copy),
        )
    }

    fn parse_place(&self, expr_id: ExprId) -> PResult<Place<'tcx>> {
        parse_by_kind!(self, expr_id, _, "place",
            ExprKind::Deref { arg } => Ok(
                self.parse_place(*arg)?.project_deeper(&[PlaceElem::Deref], self.tcx)
            ),
            _ => self.parse_local(expr_id).map(Place::from),
        )
    }

    fn parse_local(&self, expr_id: ExprId) -> PResult<Local> {
        parse_by_kind!(self, expr_id, _, "local",
            ExprKind::VarRef { id } => Ok(self.local_map[id]),
        )
    }

    fn parse_block(&self, expr_id: ExprId) -> PResult<BasicBlock> {
        parse_by_kind!(self, expr_id, _, "basic block",
            ExprKind::VarRef { id } => Ok(self.block_map[id]),
        )
    }
}
