// FIXME(#61117): Remove revisions once x86_64-gnu-debug CI job sets rust.debuginfo-level-tests=2
// NOTE: The .stderr for both revisions shall be identical.
//@ revisions: no-debuginfo full-debuginfo
//@[no-debuginfo] compile-flags: -Cdebuginfo=0
//@[full-debuginfo] compile-flags: -Cdebuginfo=2
//@ build-fail

fn generic<T: Copy>(t: T) {
    let s: [T; 1518600000] = [t; 1518600000];
    //~^ ERROR values of the type `[[u8; 1518599999]; 1518600000]` are too big
}

fn main() {
    let x: [u8; 1518599999] = [0; 1518599999];
    generic::<[u8; 1518599999]>(x);
}
