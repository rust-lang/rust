//! Metadata from source code coverage analysis and instrumentation.

use std::fmt::{self, Debug, Formatter};

use rustc_data_structures::fx::FxIndexMap;
use rustc_index::{Idx, IndexVec};
use rustc_macros::{HashStable, TyDecodable, TyEncodable};
use rustc_span::Span;

rustc_index::newtype_index! {
    /// Used by [`CoverageKind::BlockMarker`] to mark blocks during THIR-to-MIR
    /// lowering, so that those blocks can be identified later.
    #[derive(HashStable)]
    #[encodable]
    #[debug_format = "BlockMarkerId({})"]
    pub struct BlockMarkerId {}
}

rustc_index::newtype_index! {
    /// ID of a coverage counter. Values ascend from 0.
    ///
    /// Before MIR inlining, counter IDs are local to their enclosing function.
    /// After MIR inlining, coverage statements may have been inlined into
    /// another function, so use the statement's source-scope to find which
    /// function/instance its IDs are meaningful for.
    ///
    /// Note that LLVM handles counter IDs as `uint32_t`, so there is no need
    /// to use a larger representation on the Rust side.
    #[derive(HashStable)]
    #[encodable]
    #[orderable]
    #[debug_format = "CounterId({})"]
    pub struct CounterId {}
}

rustc_index::newtype_index! {
    /// ID of a coverage-counter expression. Values ascend from 0.
    ///
    /// Before MIR inlining, expression IDs are local to their enclosing function.
    /// After MIR inlining, coverage statements may have been inlined into
    /// another function, so use the statement's source-scope to find which
    /// function/instance its IDs are meaningful for.
    ///
    /// Note that LLVM handles expression IDs as `uint32_t`, so there is no need
    /// to use a larger representation on the Rust side.
    #[derive(HashStable)]
    #[encodable]
    #[orderable]
    #[debug_format = "ExpressionId({})"]
    pub struct ExpressionId {}
}

rustc_index::newtype_index! {
    /// ID of a mcdc condition. Used by llvm to check mcdc coverage.
    ///
    /// Note for future: the max limit of 0xFFFF is probably too loose. Actually llvm does not
    /// support decisions with too many conditions (7 and more at LLVM 18 while may be hundreds at 19)
    /// and represents it with `int16_t`. This max value may be changed once we could
    /// figure out an accurate limit.
    #[derive(HashStable)]
    #[encodable]
    #[orderable]
    #[max = 0xFFFF]
    #[debug_format = "ConditionId({})"]
    pub struct ConditionId {}
}

impl ConditionId {
    pub const START: Self = Self::from_usize(0);
}

/// Enum that can hold a constant zero value, the ID of an physical coverage
/// counter, or the ID of a coverage-counter expression.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[derive(TyEncodable, TyDecodable, Hash, HashStable)]
pub enum CovTerm {
    Zero,
    Counter(CounterId),
    Expression(ExpressionId),
}

impl Debug for CovTerm {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero => write!(f, "Zero"),
            Self::Counter(id) => f.debug_tuple("Counter").field(&id.as_u32()).finish(),
            Self::Expression(id) => f.debug_tuple("Expression").field(&id.as_u32()).finish(),
        }
    }
}

#[derive(Clone, PartialEq, TyEncodable, TyDecodable, Hash, HashStable)]
pub enum CoverageKind {
    /// Marks a span that might otherwise not be represented in MIR, so that
    /// coverage instrumentation can associate it with its enclosing block/BCB.
    ///
    /// Should be erased before codegen (at some point after `InstrumentCoverage`).
    SpanMarker,

    /// Marks its enclosing basic block with an ID that can be referred to by
    /// side data in [`CoverageInfoHi`].
    ///
    /// Should be erased before codegen (at some point after `InstrumentCoverage`).
    BlockMarker { id: BlockMarkerId },

    /// Marks its enclosing basic block with the ID of the coverage graph node
    /// that it was part of during the `InstrumentCoverage` MIR pass.
    ///
    /// During codegen, this might be lowered to `llvm.instrprof.increment` or
    /// to a no-op, depending on the outcome of counter-creation.
    VirtualCounter { bcb: BasicCoverageBlock },

    /// Marks the point in MIR control flow represented by a evaluated condition.
    ///
    /// This is eventually lowered to instruments updating mcdc temp variables.
    CondBitmapUpdate { index: u32, decision_depth: u16 },

    /// Marks the point in MIR control flow represented by a evaluated decision.
    ///
    /// This is eventually lowered to `llvm.instrprof.mcdc.tvbitmap.update` in LLVM IR.
    TestVectorBitmapUpdate { bitmap_idx: u32, decision_depth: u16 },
}

impl Debug for CoverageKind {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        use CoverageKind::*;
        match self {
            SpanMarker => write!(fmt, "SpanMarker"),
            BlockMarker { id } => write!(fmt, "BlockMarker({:?})", id.index()),
            VirtualCounter { bcb } => write!(fmt, "VirtualCounter({bcb:?})"),
            CondBitmapUpdate { index, decision_depth } => {
                write!(fmt, "CondBitmapUpdate(index={:?}, depth={:?})", index, decision_depth)
            }
            TestVectorBitmapUpdate { bitmap_idx, decision_depth } => {
                write!(fmt, "TestVectorUpdate({:?}, depth={:?})", bitmap_idx, decision_depth)
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, HashStable)]
#[derive(TyEncodable, TyDecodable)]
pub enum Op {
    Subtract,
    Add,
}

impl Op {
    pub fn is_add(&self) -> bool {
        matches!(self, Self::Add)
    }

    pub fn is_subtract(&self) -> bool {
        matches!(self, Self::Subtract)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[derive(TyEncodable, TyDecodable, Hash, HashStable)]
pub struct Expression {
    pub lhs: CovTerm,
    pub op: Op,
    pub rhs: CovTerm,
}

#[derive(Clone, Debug)]
#[derive(TyEncodable, TyDecodable, Hash, HashStable)]
pub enum MappingKind {
    /// Associates a normal region of code with a counter/expression/zero.
    Code { bcb: BasicCoverageBlock },
    /// Associates a branch region with separate counters for true and false.
    Branch { true_bcb: BasicCoverageBlock, false_bcb: BasicCoverageBlock },
    /// Associates a branch region with separate counters for true and false.
    MCDCBranch {
        true_bcb: BasicCoverageBlock,
        false_bcb: BasicCoverageBlock,
        mcdc_params: ConditionInfo,
    },
    /// Associates a decision region with a bitmap and number of conditions.
    MCDCDecision(DecisionInfo),
}

#[derive(Clone, Debug)]
#[derive(TyEncodable, TyDecodable, Hash, HashStable)]
pub struct Mapping {
    pub kind: MappingKind,
    pub span: Span,
}

/// Stores per-function coverage information attached to a `mir::Body`,
/// to be used in conjunction with the individual coverage statements injected
/// into the function's basic blocks.
#[derive(Clone, Debug)]
#[derive(TyEncodable, TyDecodable, Hash, HashStable)]
pub struct FunctionCoverageInfo {
    pub function_source_hash: u64,
    pub body_span: Span,

    /// Used in conjunction with `priority_list` to create physical counters
    /// and counter expressions, after MIR optimizations.
    pub node_flow_data: NodeFlowData<BasicCoverageBlock>,
    pub priority_list: Vec<BasicCoverageBlock>,

    pub mappings: Vec<Mapping>,

    pub mcdc_bitmap_bits: usize,
    /// The depth of the deepest decision is used to know how many
    /// temp condbitmaps should be allocated for the function.
    pub mcdc_num_condition_bitmaps: usize,
}

/// Coverage information for a function, recorded during MIR building and
/// attached to the corresponding `mir::Body`. Used by the `InstrumentCoverage`
/// MIR pass.
///
/// ("Hi" indicates that this is "high-level" information collected at the
/// THIR/MIR boundary, before the MIR-based coverage instrumentation pass.)
#[derive(Clone, Debug)]
#[derive(TyEncodable, TyDecodable, Hash, HashStable)]
pub struct CoverageInfoHi {
    /// 1 more than the highest-numbered [`CoverageKind::BlockMarker`] that was
    /// injected into the MIR body. This makes it possible to allocate per-ID
    /// data structures without having to scan the entire body first.
    pub num_block_markers: usize,
    pub branch_spans: Vec<BranchSpan>,
    /// Branch spans generated by mcdc. Because of some limits mcdc builder give up generating
    /// decisions including them so that they are handled as normal branch spans.
    pub mcdc_degraded_branch_spans: Vec<MCDCBranchSpan>,
    pub mcdc_spans: Vec<(MCDCDecisionSpan, Vec<MCDCBranchSpan>)>,
}

#[derive(Clone, Debug)]
#[derive(TyEncodable, TyDecodable, Hash, HashStable)]
pub struct BranchSpan {
    pub span: Span,
    pub true_marker: BlockMarkerId,
    pub false_marker: BlockMarkerId,
}

#[derive(Copy, Clone, Debug)]
#[derive(TyEncodable, TyDecodable, Hash, HashStable)]
pub struct ConditionInfo {
    pub condition_id: ConditionId,
    pub true_next_id: Option<ConditionId>,
    pub false_next_id: Option<ConditionId>,
}

#[derive(Clone, Debug)]
#[derive(TyEncodable, TyDecodable, Hash, HashStable)]
pub struct MCDCBranchSpan {
    pub span: Span,
    pub condition_info: ConditionInfo,
    pub true_marker: BlockMarkerId,
    pub false_marker: BlockMarkerId,
}

#[derive(Copy, Clone, Debug)]
#[derive(TyEncodable, TyDecodable, Hash, HashStable)]
pub struct DecisionInfo {
    pub bitmap_idx: u32,
    pub num_conditions: u16,
}

#[derive(Clone, Debug)]
#[derive(TyEncodable, TyDecodable, Hash, HashStable)]
pub struct MCDCDecisionSpan {
    pub span: Span,
    pub end_markers: Vec<BlockMarkerId>,
    pub decision_depth: u16,
    pub num_conditions: usize,
}

/// Contains information needed during codegen, obtained by inspecting the
/// function's MIR after MIR optimizations.
///
/// Returned by the `coverage_ids_info` query.
#[derive(Clone, TyEncodable, TyDecodable, Debug, HashStable)]
pub struct CoverageIdsInfo {
    pub num_counters: u32,
    pub phys_counter_for_node: FxIndexMap<BasicCoverageBlock, CounterId>,
    pub term_for_bcb: IndexVec<BasicCoverageBlock, Option<CovTerm>>,
    pub expressions: IndexVec<ExpressionId, Expression>,
}

rustc_index::newtype_index! {
    /// During the `InstrumentCoverage` MIR pass, a BCB is a node in the
    /// "coverage graph", which is a refinement of the MIR control-flow graph
    /// that merges or omits some blocks that aren't relevant to coverage.
    ///
    /// After that pass is complete, the coverage graph no longer exists, so a
    /// BCB is effectively an opaque ID.
    #[derive(HashStable)]
    #[encodable]
    #[orderable]
    #[debug_format = "bcb{}"]
    pub struct BasicCoverageBlock {
        const START_BCB = 0;
    }
}

/// Data representing a view of some underlying graph, in which each node's
/// successors have been merged into a single "supernode".
///
/// The resulting supernodes have no obvious meaning on their own.
/// However, merging successor nodes means that a node's out-edges can all
/// be combined into a single out-edge, whose flow is the same as the flow
/// (execution count) of its corresponding node in the original graph.
///
/// With all node flows now in the original graph now represented as edge flows
/// in the merged graph, it becomes possible to analyze the original node flows
/// using techniques for analyzing edge flows.
#[derive(Clone, Debug)]
#[derive(TyEncodable, TyDecodable, Hash, HashStable)]
pub struct NodeFlowData<Node: Idx> {
    /// Maps each node to the supernode that contains it, indicated by some
    /// arbitrary "root" node that is part of that supernode.
    pub supernodes: IndexVec<Node, Node>,
    /// For each node, stores the single supernode that all of its successors
    /// have been merged into.
    ///
    /// (Note that each node in a supernode can potentially have a _different_
    /// successor supernode from its peers.)
    pub succ_supernodes: IndexVec<Node, Node>,
}
