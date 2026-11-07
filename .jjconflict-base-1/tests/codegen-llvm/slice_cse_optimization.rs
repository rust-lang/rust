//! Various iterating method over slice correctly optimized using common subexpression elimination.
//! Checks function has memory(argmem: read) attribute.
//! Regression test for <https://github.com/rust-lang/rust/issues/119573>.
//@ compile-flags: -O

#![crate_type = "lib"]
// CHECK-LABEL: @has_zero_iter
// CHECK-SAME: #[[ATTR:[0-9]+]]
#[inline(never)]
#[unsafe(no_mangle)]
pub fn has_zero_iter(xs: &[u8]) -> bool {
    xs.iter().any(|&x| x == 0)
}

// CHECK-LABEL: @has_zero_ptr
// CHECK-SAME: #[[ATTR]]
#[inline(never)]
#[unsafe(no_mangle)]
fn has_zero_ptr(xs: &[u8]) -> bool {
    let range = xs.as_ptr_range();
    let mut start = range.start;
    let end = range.end;
    while start < end {
        unsafe {
            if *start == 0 {
                return true;
            }
            start = start.add(1);
        }
    }
    false
}
// CHECK-LABEL: @has_zero_for
// CHECK-SAME: #[[ATTR]]
#[inline(never)]
#[unsafe(no_mangle)]
fn has_zero_for(xs: &[u8]) -> bool {
    for x in xs {
        if *x == 0 {
            return true;
        }
    }
    false
}

// CHECK: attributes #[[ATTR]] = { {{.*}}memory(argmem: read){{.*}} }
