pub use crate::very_unstable::hir::ImplicitSelfKind;
pub use crate::very_unstable::middle::mir::{
    visit::MutVisitor, AggregateKind, AssertKind, BasicBlock, BasicBlockData, BinOp, BindingForm,
    BlockTailInfo, Body, BorrowKind, CastKind, ClearCrossCrate, Constant, ConstantKind,
    CopyNonOverlapping, Coverage, FakeReadCause, Field, GeneratorInfo, InlineAsmOperand, Local,
    LocalDecl, LocalInfo, LocalKind, Location, MirPhase, MirSource, NullOp, Operand, Place,
    PlaceRef, ProjectionElem, ProjectionKind, Promoted, RetagKind, Rvalue, Safety, SourceInfo,
    SourceScope, SourceScopeData, SourceScopeLocalData, Statement, StatementKind, UnOp,
    UserTypeProjection, UserTypeProjections, VarBindingForm, VarDebugInfo, VarDebugInfoContents,
};
