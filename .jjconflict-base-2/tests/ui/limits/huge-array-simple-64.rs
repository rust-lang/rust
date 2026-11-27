// FIXME(#61117): Remove revisions once x86_64-gnu-debug CI job sets rust.debuginfo-level-tests=2
// NOTE: The .stderr for both revisions shall be identical.
//@ revisions: no-debuginfo full-debuginfo
//@[no-debuginfo] compile-flags: -Cdebuginfo=0
//@[full-debuginfo] compile-flags: -Cdebuginfo=2
//@ build-fail
//@ ignore-32bit

#![allow(arithmetic_overflow)]

fn main() {
    let _fat: [u8; (1<<61)+(1<<31)] = //~ ERROR too big for the target architecture
        [0; (1u64<<61) as usize +(1u64<<31) as usize];
}
