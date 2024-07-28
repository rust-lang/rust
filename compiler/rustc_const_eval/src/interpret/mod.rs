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

use eval_context::{from_known_layout, mir_assign_valid_types};
#[doc(no_inline)]
pub use rustc_middle::mir::interpret::*; // have all the `interpret` symbols in one place: here

pub use self::eval_context::{format_interp_error, Frame, FrameInfo, InterpCx, StackPopCleanup};
pub use self::intern::{
    intern_const_alloc_for_constprop, intern_const_alloc_recursive, HasStaticRootDefId, InternKind,
    InternResult,
};
pub(crate) use self::intrinsics::eval_nullary_intrinsic;
pub use self::machine::{compile_time_machine, AllocMap, Machine, MayLeak, ReturnAction};
pub use self::memory::{AllocKind, AllocRef, AllocRefMut, FnVal, Memory, MemoryKind};
use self::operand::Operand;
pub use self::operand::{ImmTy, Immediate, OpTy, Readable};
pub use self::place::{MPlaceTy, MemPlaceMeta, PlaceTy, Writeable};
use self::place::{MemPlace, Place};
pub use self::projection::{OffsetMode, Projectable};
pub use self::terminator::FnArg;
pub(crate) use self::util::create_static_alloc;
pub use self::validity::{CtfeValidationMode, RefTracking};
pub use self::visitor::ValueVisitor;
