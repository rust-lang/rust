//! An interpreter for MIR used in CTFE and by miri

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
mod step;
mod terminator;
mod traits;
mod util;
mod validity;
mod visitor;

pub use rustc_middle::mir::interpret::*; // have all the `interpret` symbols in one place: here

pub use self::eval_context::{
    Frame, FrameInfo, InterpCx, LocalState, LocalValue, StackPopCleanup, StackPopUnwind,
};
pub use self::intern::{intern_const_alloc_recursive, InternKind};
pub use self::machine::{compile_time_machine, AllocMap, Machine, MayLeak, StackPopJump};
pub use self::memory::{AllocKind, AllocRef, AllocRefMut, FnVal, Memory, MemoryKind};
pub use self::operand::{ImmTy, Immediate, OpTy, Operand};
pub use self::place::{MPlaceTy, MemPlace, MemPlaceMeta, Place, PlaceTy};
pub use self::validity::{CtfeValidationMode, RefTracking};
pub use self::visitor::{MutValueVisitor, Value, ValueVisitor};

pub(crate) use self::intrinsics::eval_nullary_intrinsic;
use eval_context::{from_known_layout, mir_assign_valid_types};
