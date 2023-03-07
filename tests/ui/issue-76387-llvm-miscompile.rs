// no-system-llvm
// compile-flags: -C opt-level=3
// aux-build: issue-76387.rs
// run-pass

// Regression test for issue #76387
// Tests that LLVM doesn't miscompile this

extern crate issue_76387;

use issue_76387::FatPtr;

fn print(data: &[u8]) {
    println!("{:#?}", data);
}

fn main() {
    let ptr = FatPtr::new(20);
    let data = unsafe { std::slice::from_raw_parts(ptr.as_ptr(), ptr.len()) };

    print(data);
}
