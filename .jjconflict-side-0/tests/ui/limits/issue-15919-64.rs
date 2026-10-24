// FIXME(#61117): Remove revisions once x86_64-gnu-debug CI job sets rust.debuginfo-level-tests=2
// NOTE: The .stderr for both revisions shall be identical.
//@ revisions: no-debuginfo full-debuginfo
//@[no-debuginfo] compile-flags: -Cdebuginfo=0
//@[full-debuginfo] compile-flags: -Cdebuginfo=2
//@ build-fail
//@ ignore-32bit

fn main() {
    let x = [0usize; 0xffff_ffff_ffff_ffff]; //~ ERROR too big
}

// This and the -32 version of this test need to have different literals, as we can't rely on
// conditional compilation for them while retaining the same spans/lines.
