// This file is imported into every module by default.

/* Reexported core operators */

pub use kinds::{Const, Copy, Owned, Durable};
pub use ops::{Drop};
pub use ops::{Add, Sub, Mul, Div, Modulo, Neg};
pub use ops::{BitAnd, BitOr, BitXor};
pub use ops::{Shl, Shr, Index};
pub use option::{Option, Some, None};
pub use result::{Result, Ok, Err};

/* Reexported types and traits */

pub use path::Path;
pub use path::GenericPath;
pub use path::WindowsPath;
pub use path::PosixPath;

pub use tuple::{CopyableTuple, ImmutableTuple, ExtendedTupleOps};
pub use str::{StrSlice, Trimmable};
pub use vec::{ConstVector, CopyableVector, ImmutableVector};
pub use vec::{ImmutableEqVector, ImmutableCopyableVector};
pub use vec::{MutableVector, MutableCopyableVector};
pub use iter::{BaseIter, ExtendedIter, EqIter, CopyableIter};
pub use iter::{CopyableOrderedIter, CopyableNonstrictIter, Times};

pub use num::Num;
pub use ptr::Ptr;
pub use to_str::ToStr;
pub use clone::Clone;

pub use cmp::{Eq, Ord};
pub use hash::Hash;
pub use to_bytes::IterBytes;

/* Reexported modules */

pub use at_vec;
pub use bool;
pub use cast;
pub use char;
pub use cmp;
pub use dvec;
pub use either;
pub use extfmt;
pub use f32;
pub use f64;
pub use float;
pub use i16;
pub use i32;
pub use i64;
pub use i8;
pub use int;
pub use io;
pub use iter;
pub use libc;
pub use num;
pub use oldcomm;
pub use ops;
pub use option;
pub use os;
pub use path;
pub use pipes;
pub use private;
pub use ptr;
pub use rand;
pub use result;
pub use str;
pub use sys;
pub use task;
pub use to_str;
pub use u16;
pub use u32;
pub use u64;
pub use u8;
pub use uint;
pub use vec;

/*
 * Export the log levels as global constants. Higher levels mean
 * more-verbosity. Error is the bottom level, default logging level is
 * warn-and-below.
 */

/// The error log level
pub const error : u32 = 1_u32;
/// The warning log level
pub const warn : u32 = 2_u32;
/// The info log level
pub const info : u32 = 3_u32;
/// The debug log level
pub const debug : u32 = 4_u32;
