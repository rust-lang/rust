//! An interpreter for MIR used in CTFE and by miri

mod call;
mod cast;
mod discriminant;
mod eval_context;
mod intern;
mod intrinsics;
mod machine;
mod memory;
mod operand;
mod operator;
mod place;
mod projection;
mod stack;
mod step;
mod traits;
mod util;
mod validity;
mod visitor;

#[doc(no_inline)]
pub use rustc_middle::mir::interpret::*; // have all the `interpret` symbols in one place: here

pub use self::call::FnArg;
pub use self::eval_context::{InterpCx, format_interp_error};
use self::eval_context::{from_known_layout, mir_assign_valid_types};
pub use self::intern::{
    HasStaticRootDefId, InternKind, InternResult, intern_const_alloc_for_constprop,
    intern_const_alloc_recursive,
};
pub use self::machine::{AllocMap, Machine, MayLeak, ReturnAction, compile_time_machine};
pub use self::memory::{AllocInfo, AllocKind, AllocRef, AllocRefMut, FnVal, Memory, MemoryKind};
use self::operand::Operand;
pub use self::operand::{ImmTy, Immediate, OpTy};
pub use self::place::{MPlaceTy, MemPlaceMeta, PlaceTy, Writeable};
use self::place::{MemPlace, Place};
pub use self::projection::{OffsetMode, Projectable};
pub use self::stack::{Frame, FrameInfo, LocalState, StackPopCleanup, StackPopInfo};
pub(crate) use self::util::create_static_alloc;
pub use self::validity::{CtfeValidationMode, RangeSet, RefTracking};
pub use self::visitor::ValueVisitor;
