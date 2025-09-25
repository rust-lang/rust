// FIXME(#61117): Remove revisions once x86_64-gnu-debug CI job sets rust.debuginfo-level-tests=2
// NOTE: The .stderr for both revisions shall be identical.
//@ revisions: no-debuginfo full-debuginfo
//@ build-fail
//@ normalize-stderr: "std::option::Option<\[u32; \d+\]>" -> "TYPE"
//@ normalize-stderr: "\[u32; \d+\]" -> "TYPE"
//@[no-debuginfo] compile-flags: -Cdebuginfo=0
//@[full-debuginfo] compile-flags: -Cdebuginfo=2

#[cfg(target_pointer_width = "32")]
type BIG = Option<[u32; (1<<29)-1]>;

#[cfg(target_pointer_width = "64")]
type BIG = Option<[u32; (1<<59)-1]>;

fn main() {
    let big: BIG = None;
    //~^ ERROR are too big for the target architecture
}
