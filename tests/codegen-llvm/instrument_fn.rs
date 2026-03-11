// Verify the #[instrument_fn] applies the correct LLVM IR function attributes.
//
//@ revisions:XRAY MCOUNT FENTRY
//@ add-minicore
//@ compile-flags: -Copt-level=0
//@ [XRAY] compile-flags: -Zinstrument-function=xray
//@ [MCOUNT] compile-flags: -Zinstrument-function=mcount
//@ [FENTRY] compile-flags: -Zinstrument-function=fentry --target=x86_64-unknown-linux-gnu
//@ [FENTRY] needs-llvm-components: x86

#![feature(no_core)]
#![crate_type = "lib"]
#![feature(instrument_fn)]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
// CHECK: define void @instrument_default() {{.*}} [[DFLT_ATTR:#[0-9]+]]
fn instrument_default() {}

#[no_mangle]
#[instrument_fn = "off"]
// CHECK: define void @instrument_off() {{.*}} [[OFF_ATTR:#[0-9]+]]
fn instrument_off() {}

#[no_mangle]
#[instrument_fn = "on"]
// MCOUNT: define void @instrument_on() {{.*}} [[DFLT_ATTR]]
// FENTRY: define void @instrument_on() {{.*}} [[DFLT_ATTR]]
// XRAY: define void @instrument_on() {{.*}} [[ON_ATTR:#[0-9]+]]
fn instrument_on() {}

// MCOUNT: attributes [[DFLT_ATTR]] {{.*}} "instrument-function-entry-inlined"=
// MCOUNT-NOT: attributes [[OFF_ATTR]] {{.*}} "instrument-function-entry-inlined"=

// FENTRY: attributes [[DFLT_ATTR]] {{.*}} "fentry-call"="true"
// FENTRY-NOT: attributes [[OFF_ATTR]] {{.*}} "fentry-call"="true"

// XRAY-NOT: attributes [[DFLT_ATTR]] {{.*}} "xray-skip-exit"
// XRAY-NOT: attributes [[DFLT_ATTR]] {{.*}} "xray-skip-entry"

// XRAY-NOT: attributes [[OFF_ATTR]] {{.*}} "function-instrument"="xray-always"
// XRAY:     attributes [[OFF_ATTR]] {{.*}} "function-instrument"="xray-never"
// XRAY-NOT: attributes [[OFF_ATTR]] {{.*}} "xray-skip-exit"
// XRAY-NOT: attributes [[OFF_ATTR]] {{.*}} "xray-skip-entry"

// XRAY:     attributes [[ON_ATTR]] {{.*}} "function-instrument"="xray-always"
// XRAY-NOT: attributes [[ON_ATTR]] {{.*}} "function-instrument"="xray-never"
// XRAY-NOT: attributes [[ON_ATTR]] {{.*}} "xray-skip-exit"
// XRAY-NOT: attributes [[ON_ATTR]] {{.*}} "xray-skip-entry"
