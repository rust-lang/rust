use super::debug;

use debug::DebugCounters;

use rustc_middle::mir::coverage::*;

/// Manages the counter and expression indexes/IDs to generate `CoverageKind` components for MIR
/// `Coverage` statements.
pub(crate) struct CoverageCounters {
    function_source_hash: u64,
    next_counter_id: u32,
    num_expressions: u32,
    pub debug_counters: DebugCounters,
}

impl CoverageCounters {
    pub fn new(function_source_hash: u64) -> Self {
        Self {
            function_source_hash,
            next_counter_id: CounterValueReference::START.as_u32(),
            num_expressions: 0,
            debug_counters: DebugCounters::new(),
        }
    }

    /// Activate the `DebugCounters` data structures, to provide additional debug formatting
    /// features when formating `CoverageKind` (counter) values.
    pub fn enable_debug(&mut self) {
        self.debug_counters.enable();
    }

    pub fn make_counter<F>(&mut self, debug_block_label_fn: F) -> CoverageKind
    where
        F: Fn() -> Option<String>,
    {
        let counter = CoverageKind::Counter {
            function_source_hash: self.function_source_hash,
            id: self.next_counter(),
        };
        if self.debug_counters.is_enabled() {
            self.debug_counters.add_counter(&counter, (debug_block_label_fn)());
        }
        counter
    }

    pub fn make_expression<F>(
        &mut self,
        lhs: ExpressionOperandId,
        op: Op,
        rhs: ExpressionOperandId,
        debug_block_label_fn: F,
    ) -> CoverageKind
    where
        F: Fn() -> Option<String>,
    {
        let id = self.next_expression();
        let expression = CoverageKind::Expression { id, lhs, op, rhs };
        if self.debug_counters.is_enabled() {
            self.debug_counters.add_counter(&expression, (debug_block_label_fn)());
        }
        expression
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
