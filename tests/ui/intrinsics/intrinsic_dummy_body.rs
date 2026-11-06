//@ build-pass

#![feature(core_intrinsics)]
#![expect(internal_features)]

use std::intrinsics::abort_immediate as main;
