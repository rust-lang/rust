// compile-flags: -O
#![crate_type = "lib"]

// Test that LLVM can eliminate the unreachable `All::None` branch.

pub enum All {
    None,
    Foo,
    Bar,
}

// CHECK-LABEL: @issue_73031
#[no_mangle]
pub fn issue_73031(a: &mut All, q: i32) -> i32 {
    *a = if q == 5 {
        All::Foo
    } else {
        All::Bar
    };
    match *a {
        // CHECK-NOT: panic
        All::None => panic!(),
        All::Foo => 1,
        All::Bar => 2,
    }
}
