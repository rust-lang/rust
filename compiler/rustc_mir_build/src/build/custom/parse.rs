use rustc_index::IndexSlice;
use rustc_middle::{mir::*, thir::*, ty::Ty};
use rustc_span::Span;

use super::{PResult, ParseCtxt, ParseError};

mod instruction;

/// Helper macro for parsing custom MIR.
///
/// Example usage looks something like:
/// ```rust,ignore (incomplete example)
/// parse_by_kind!(
///     self, // : &ParseCtxt
///     expr_id, // what you're matching against
///     "assignment", // the thing you're trying to parse
///     @call("mir_assign", args) => { args[0] }, // match invocations of the `mir_assign` special function
///     ExprKind::Assign { lhs, .. } => { lhs }, // match thir assignment expressions
///     // no need for fallthrough case - reasonable error is automatically generated
/// )
/// ```
macro_rules! parse_by_kind {
    (
        $self:ident,
        $expr_id:expr,
        $expr_name:pat,
        $expected:literal,
        $(
            @call($name:literal, $args:ident) => $call_expr:expr,
        )*
        $(
            $pat:pat => $expr:expr,
        )*
    ) => {{
        let expr_id = $self.preparse($expr_id);
        let expr = &$self.thir[expr_id];
        debug!("Trying to parse {:?} as {}", expr.kind, $expected);
        let $expr_name = expr;
        match &expr.kind {
            $(
                ExprKind::Call { ty, fun: _, args: $args, .. } if {
                    match ty.kind() {
                        ty::FnDef(did, _) => {
                            $self.tcx.is_diagnostic_item(rustc_span::Symbol::intern($name), *did)
                        }
                        _ => false,
                    }
                } => $call_expr,
            )*
            $(
                $pat => $expr,
            )*
            #[allow(unreachable_patterns)]
            _ => return Err($self.expr_error(expr_id, $expected))
        }
    }};
}
pub(crate) use parse_by_kind;

impl<'tcx, 'body> ParseCtxt<'tcx, 'body> {
    /// Expressions should only ever be matched on after preparsing them. This removes extra scopes
    /// we don't care about.
    fn preparse(&self, expr_id: ExprId) -> ExprId {
        let expr = &self.thir[expr_id];
        match expr.kind {
            ExprKind::Scope { value, .. } => self.preparse(value),
            _ => expr_id,
        }
    }

    fn statement_as_expr(&self, stmt_id: StmtId) -> PResult<ExprId> {
        match &self.thir[stmt_id].kind {
            StmtKind::Expr { expr, .. } => Ok(*expr),
            kind @ StmtKind::Let { pattern, .. } => {
                return Err(ParseError {
                    span: pattern.span,
                    item_description: format!("{:?}", kind),
                    expected: "expression".to_string(),
                });
            }
        }
    }

    pub fn parse_args(&mut self, params: &IndexSlice<ParamId, Param<'tcx>>) -> PResult<()> {
        for param in params.iter() {
            let (var, span) = {
                let pat = param.pat.as_ref().unwrap();
                match &pat.kind {
                    PatKind::Binding { var, .. } => (*var, pat.span),
                    _ => {
                        return Err(ParseError {
                            span: pat.span,
                            item_description: format!("{:?}", pat.kind),
                            expected: "local".to_string(),
                        });
                    }
                }
            };
            let decl = LocalDecl::new(param.ty, span);
            let local = self.body.local_decls.push(decl);
            self.local_map.insert(var, local);
        }

        Ok(())
    }

    /// Bodies are of the form:
    ///
    /// ```text
    /// {
    ///     let bb1: BasicBlock;
    ///     let bb2: BasicBlock;
    ///     {
    ///         let RET: _;
    ///         let local1;
    ///         let local2;
    ///
    ///         {
    ///             { // entry block
    ///                 statement1;
    ///                 terminator1
    ///             };
    ///
    ///             bb1 = {
    ///                 statement2;
    ///                 terminator2
    ///             };
    ///
    ///             bb2 = {
    ///                 statement3;
    ///                 terminator3
    ///             }
    ///
    ///             RET
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// This allows us to easily parse the basic blocks declarations, local declarations, and
    /// basic block definitions in order.
    pub fn parse_body(&mut self, expr_id: ExprId) -> PResult<()> {
        let body = parse_by_kind!(self, expr_id, _, "whole body",
            ExprKind::Block { block } => self.thir[*block].expr.unwrap(),
        );
        let (block_decls, rest) = parse_by_kind!(self, body, _, "body with block decls",
            ExprKind::Block { block } => {
                let block = &self.thir[*block];
                (&block.stmts, block.expr.unwrap())
            },
        );
        self.parse_block_decls(block_decls.iter().copied())?;

        let (local_decls, rest) = parse_by_kind!(self, rest, _, "body with local decls",
            ExprKind::Block { block } => {
                let block = &self.thir[*block];
                (&block.stmts, block.expr.unwrap())
            },
        );
        self.parse_local_decls(local_decls.iter().copied())?;

        let block_defs = parse_by_kind!(self, rest, _, "body with block defs",
            ExprKind::Block { block } => &self.thir[*block].stmts,
        );
        for (i, block_def) in block_defs.iter().enumerate() {
            let block = self.parse_block_def(self.statement_as_expr(*block_def)?)?;
            self.body.basic_blocks_mut()[BasicBlock::from_usize(i)] = block;
        }

        Ok(())
    }

    fn parse_block_decls(&mut self, stmts: impl Iterator<Item = StmtId>) -> PResult<()> {
        for stmt in stmts {
            let (var, _, _) = self.parse_let_statement(stmt)?;
            let data = BasicBlockData::new(None);
            let block = self.body.basic_blocks_mut().push(data);
            self.block_map.insert(var, block);
        }

        Ok(())
    }

    fn parse_local_decls(&mut self, mut stmts: impl Iterator<Item = StmtId>) -> PResult<()> {
        let (ret_var, ..) = self.parse_let_statement(stmts.next().unwrap())?;
        self.local_map.insert(ret_var, Local::from_u32(0));

        for stmt in stmts {
            let (var, ty, span) = self.parse_let_statement(stmt)?;
            let decl = LocalDecl::new(ty, span);
            let local = self.body.local_decls.push(decl);
            self.local_map.insert(var, local);
        }

        Ok(())
    }

    fn parse_let_statement(&mut self, stmt_id: StmtId) -> PResult<(LocalVarId, Ty<'tcx>, Span)> {
        let pattern = match &self.thir[stmt_id].kind {
            StmtKind::Let { pattern, .. } => pattern,
            StmtKind::Expr { expr, .. } => {
                return Err(self.expr_error(*expr, "let statement"));
            }
        };

        self.parse_var(pattern)
    }

    fn parse_var(&mut self, mut pat: &Pat<'tcx>) -> PResult<(LocalVarId, Ty<'tcx>, Span)> {
        // Make sure we throw out any `AscribeUserType` we find
        loop {
            match &pat.kind {
                PatKind::Binding { var, ty, .. } => break Ok((*var, *ty, pat.span)),
                PatKind::AscribeUserType { subpattern, .. } => {
                    pat = subpattern;
                }
                _ => {
                    break Err(ParseError {
                        span: pat.span,
                        item_description: format!("{:?}", pat.kind),
                        expected: "local".to_string(),
                    });
                }
            }
        }
    }

    fn parse_block_def(&self, expr_id: ExprId) -> PResult<BasicBlockData<'tcx>> {
        let block = parse_by_kind!(self, expr_id, _, "basic block",
            ExprKind::Block { block } => &self.thir[*block],
        );

        let mut data = BasicBlockData::new(None);
        for stmt_id in &*block.stmts {
            let stmt = self.statement_as_expr(*stmt_id)?;
            let span = self.thir[stmt].span;
            let statement = self.parse_statement(stmt)?;
            data.statements.push(Statement {
                source_info: SourceInfo { span, scope: self.source_scope },
                kind: statement,
            });
        }

        let Some(trailing) = block.expr else { return Err(self.expr_error(expr_id, "terminator")) };
        let span = self.thir[trailing].span;
        let terminator = self.parse_terminator(trailing)?;
        data.terminator = Some(Terminator {
            source_info: SourceInfo { span, scope: self.source_scope },
            kind: terminator,
        });

        Ok(data)
    }
}
