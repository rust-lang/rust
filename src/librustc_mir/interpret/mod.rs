// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An interpreter for MIR used in CTFE and by miri

mod cast;
mod eval_context;
mod place;
mod operand;
mod machine;
mod memory;
mod operator;
pub(crate) mod snapshot; // for const_eval
mod step;
mod terminator;
mod traits;
mod validity;
mod intrinsics;

pub use self::eval_context::{
    EvalContext, Frame, StackPopCleanup, LocalValue,
};

pub use self::place::{Place, PlaceTy, MemPlace, MPlaceTy};

pub use self::memory::{Memory, MemoryKind};

pub use self::machine::{Machine, AllocMap};

pub use self::operand::{ScalarMaybeUndef, Value, ValTy, Operand, OpTy};

pub use self::validity::RefTracking;
