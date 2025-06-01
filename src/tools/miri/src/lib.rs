#![feature(rustc_private)]
#![feature(cfg_select)]
#![feature(float_gamma)]
#![feature(float_erf)]
#![feature(map_try_insert)]
#![feature(never_type)]
#![feature(try_blocks)]
#![feature(io_error_more)]
#![feature(variant_count)]
#![feature(yeet_expr)]
#![feature(nonzero_ops)]
#![feature(strict_overflow_ops)]
#![feature(pointer_is_aligned_to)]
#![feature(ptr_metadata)]
#![feature(unqualified_local_imports)]
#![feature(derive_coerce_pointee)]
#![feature(arbitrary_self_types)]
// Configure clippy and other lints
#![allow(
    clippy::collapsible_else_if,
    clippy::collapsible_if,
    clippy::if_same_then_else,
    clippy::comparison_chain,
    clippy::enum_variant_names,
    clippy::field_reassign_with_default,
    clippy::manual_map,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::new_without_default,
    clippy::single_match,
    clippy::useless_format,
    clippy::derive_partial_eq_without_eq,
    clippy::derived_hash_with_manual_eq,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::bool_to_int_with_if,
    clippy::needless_question_mark,
    clippy::needless_lifetimes,
    clippy::too_long_first_doc_paragraph,
    // We don't use translatable diagnostics
    rustc::diagnostic_outside_of_impl,
    // We are not implementing queries here so it's fine
    rustc::potential_query_instability,
    rustc::untranslatable_diagnostic,
)]
#![warn(rust_2018_idioms, unqualified_local_imports, clippy::as_conversions)]
// Needed for rustdoc from bootstrap (with `-Znormalize-docs`).
#![recursion_limit = "256"]

// Some "regular" crates we want to share with rustc
extern crate either;
extern crate tracing;

// The rustc crates we need
extern crate rustc_abi;
extern crate rustc_apfloat;
extern crate rustc_ast;
extern crate rustc_attr_data_structures;
extern crate rustc_const_eval;
extern crate rustc_data_structures;
extern crate rustc_errors;
extern crate rustc_hir;
extern crate rustc_index;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_symbol_mangling;
extern crate rustc_target;
// Linking `rustc_driver` pulls in the required  object code as the rest of the rustc crates are
// shipped only as rmeta files.
#[allow(unused_extern_crates)]
extern crate rustc_driver;

mod alloc;
mod alloc_addresses;
mod borrow_tracker;
mod clock;
mod concurrency;
mod diagnostics;
mod eval;
mod helpers;
mod intrinsics;
mod machine;
mod math;
mod mono_hash_map;
mod operator;
mod provenance_gc;
mod range_map;
mod shims;

// Establish a "crate-wide prelude": we often import `crate::*`.
// Make all those symbols available in the same place as our own.
#[doc(no_inline)]
pub use rustc_const_eval::interpret::*;
// Resolve ambiguity.
#[doc(no_inline)]
pub use rustc_const_eval::interpret::{self, AllocMap, Provenance as _};
use rustc_middle::{bug, span_bug};
use tracing::{info, trace};

// Type aliases that set the provenance parameter.
pub type Pointer = interpret::Pointer<Option<machine::Provenance>>;
pub type StrictPointer = interpret::Pointer<machine::Provenance>;
pub type Scalar = interpret::Scalar<machine::Provenance>;
pub type ImmTy<'tcx> = interpret::ImmTy<'tcx, machine::Provenance>;
pub type OpTy<'tcx> = interpret::OpTy<'tcx, machine::Provenance>;
pub type PlaceTy<'tcx> = interpret::PlaceTy<'tcx, machine::Provenance>;
pub type MPlaceTy<'tcx> = interpret::MPlaceTy<'tcx, machine::Provenance>;

pub use crate::alloc::MiriAllocBytes;
pub use crate::alloc_addresses::{EvalContextExt as _, ProvenanceMode};
pub use crate::borrow_tracker::stacked_borrows::{
    EvalContextExt as _, Item, Permission, Stack, Stacks,
};
pub use crate::borrow_tracker::tree_borrows::{EvalContextExt as _, Tree};
pub use crate::borrow_tracker::{BorTag, BorrowTrackerMethod, EvalContextExt as _, RetagFields};
pub use crate::clock::{Instant, MonotonicClock};
pub use crate::concurrency::cpu_affinity::MAX_CPUS;
pub use crate::concurrency::data_race::{
    AtomicFenceOrd, AtomicReadOrd, AtomicRwOrd, AtomicWriteOrd, EvalContextExt as _,
};
pub use crate::concurrency::init_once::{EvalContextExt as _, InitOnceId};
pub use crate::concurrency::sync::{
    CondvarId, EvalContextExt as _, MutexRef, RwLockId, SynchronizationObjects,
};
pub use crate::concurrency::thread::{
    BlockReason, DynUnblockCallback, EvalContextExt as _, StackEmptyCallback, ThreadId,
    ThreadManager, TimeoutAnchor, TimeoutClock, UnblockKind,
};
pub use crate::concurrency::{GenmcConfig, GenmcCtx};
pub use crate::diagnostics::{
    EvalContextExt as _, NonHaltingDiagnostic, TerminationInfo, report_error,
};
pub use crate::eval::{
    AlignmentCheck, BacktraceStyle, IsolatedOp, MiriConfig, MiriEntryFnType, RejectOpWith,
    ValidationMode, create_ecx, eval_entry,
};
pub use crate::helpers::{AccessKind, EvalContextExt as _, ToU64 as _, ToUsize as _};
pub use crate::intrinsics::EvalContextExt as _;
pub use crate::machine::{
    AllocExtra, DynMachineCallback, FrameExtra, MachineCallback, MemoryKind, MiriInterpCx,
    MiriInterpCxExt, MiriMachine, MiriMemoryKind, PrimitiveLayouts, Provenance, ProvenanceExtra,
};
pub use crate::mono_hash_map::MonoHashMap;
pub use crate::operator::EvalContextExt as _;
pub use crate::provenance_gc::{EvalContextExt as _, LiveAllocs, VisitProvenance, VisitWith};
pub use crate::range_map::RangeMap;
pub use crate::shims::EmulateItemResult;
pub use crate::shims::env::{EnvVars, EvalContextExt as _};
pub use crate::shims::foreign_items::{DynSym, EvalContextExt as _};
pub use crate::shims::io_error::{EvalContextExt as _, IoError, LibcError};
pub use crate::shims::os_str::EvalContextExt as _;
pub use crate::shims::panic::{CatchUnwindData, EvalContextExt as _};
pub use crate::shims::time::EvalContextExt as _;
pub use crate::shims::tls::TlsData;

/// Insert rustc arguments at the beginning of the argument list that Miri wants to be
/// set per default, for maximal validation power.
/// Also disable the MIR pass that inserts an alignment check on every pointer dereference. Miri
/// does that too, and with a better error message.
pub const MIRI_DEFAULT_ARGS: &[&str] = &[
    "--cfg=miri",
    "-Zalways-encode-mir",
    "-Zextra-const-ub-checks",
    "-Zmir-emit-retag",
    "-Zmir-preserve-ub",
    "-Zmir-opt-level=0",
    "-Zmir-enable-passes=-CheckAlignment,-CheckNull",
    // Deduplicating diagnostics means we miss events when tracking what happens during an
    // execution. Let's not do that.
    "-Zdeduplicate-diagnostics=no",
];
