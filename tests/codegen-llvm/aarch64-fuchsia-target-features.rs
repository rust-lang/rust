//@ add-minicore
//@ revisions: DEFAULT GENERIC A53 A73 A73_ENABLE DEFAULT_DISABLE
//@ compile-flags: --crate-type=rlib --target=aarch64-unknown-fuchsia
//@ needs-llvm-components: aarch64
//@ [GENERIC] compile-flags: -C target-cpu=generic
//@ [A53] compile-flags: -C target-cpu=cortex-a53
//@ [A73] compile-flags: -C target-cpu=cortex-a73
//@ [A73_ENABLE] compile-flags: -C target-cpu=cortex-a73 -C target-feature=+fix-cortex-a53-835769
//@ [DEFAULT_DISABLE] compile-flags: -C target-feature=-fix-cortex-a53-835769

// DEFAULT: attributes #0 = { {{.*}}"target-features"="{{.*}}+fix-cortex-a53-835769{{.*}}" }
// GENERIC: attributes #0 = { {{.*}}"target-features"="{{.*}}+fix-cortex-a53-835769{{.*}}" }
// A53: attributes #0 = { {{.*}}"target-features"="{{.*}}+fix-cortex-a53-835769{{.*}}" }
// A73-NOT: fix-cortex-a53-835769
// A73_ENABLE: attributes #0 = { {{.*}}"target-features"="{{.*}}+fix-cortex-a53-835769{{.*}}" }
// DEFAULT_DISABLE: attributes #0 = { {{.*}}"target-features"="{{.*}}-fix-cortex-a53-835769" }

#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub fn test() {}
