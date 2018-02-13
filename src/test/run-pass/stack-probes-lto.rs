// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-arm
// ignore-aarch64
// ignore-wasm
// ignore-cloudabi no processes
// ignore-emscripten no processes
// ignore-musl FIXME #31506
// ignore-pretty
// no-system-llvm
// compile-flags: -C lto
// no-prefer-dynamic

include!("stack-probes.rs");
