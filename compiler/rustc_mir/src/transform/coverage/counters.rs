use rustc_middle::mir::coverage::*;

/// Manages the counter and expression indexes/IDs to generate `CoverageKind` components for MIR
/// `Coverage` statements.
pub(crate) struct CoverageCounters {
    function_source_hash: u64,
    next_counter_id: u32,
    num_expressions: u32,
}

impl CoverageCounters {
    pub fn new(function_source_hash: u64) -> Self {
        Self {
            function_source_hash,
            next_counter_id: CounterValueReference::START.as_u32(),
            num_expressions: 0,
        }
    }

    pub fn make_counter(&mut self) -> CoverageKind {
        CoverageKind::Counter {
            function_source_hash: self.function_source_hash,
            id: self.next_counter(),
        }
    }

    pub fn make_expression(
        &mut self,
        lhs: ExpressionOperandId,
        op: Op,
        rhs: ExpressionOperandId,
    ) -> CoverageKind {
        let id = self.next_expression();
        CoverageKind::Expression { id, lhs, op, rhs }
    }

    /// Counter IDs start from one and go up.
    fn next_counter(&mut self) -> CounterValueReference {
        assert!(self.next_counter_id < u32::MAX - self.num_expressions);
        let next = self.next_counter_id;
        self.next_counter_id += 1;
        CounterValueReference::from(next)
    }

    /// Expression IDs start from u32::MAX and go down because a Expression can reference
    /// (add or subtract counts) of both Counter regions and Expression regions. The counter
    /// expression operand IDs must be unique across both types.
    fn next_expression(&mut self) -> InjectedExpressionId {
        assert!(self.next_counter_id < u32::MAX - self.num_expressions);
        let next = u32::MAX - self.num_expressions;
        self.num_expressions += 1;
        InjectedExpressionId::from(next)
    }
}
