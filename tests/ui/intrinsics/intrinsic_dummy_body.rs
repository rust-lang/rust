//@ build-fail
//~? ERROR: missing optimized MIR for `abort_immediate` in the crate `core`

#![feature(core_intrinsics)]
#![expect(internal_features)]

use std::intrinsics::abort_immediate as main;
