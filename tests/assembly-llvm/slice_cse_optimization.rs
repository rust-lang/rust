//! Various iterating method over slice correctly optimized using common subexpression elimination.
//! Regression test for <https://github.com/rust-lang/rust/issues/119573>.
//@ assembly-output: emit-asm
//@ compile-flags: -O
//@ only-x86_64

#![crate_type = "lib"]
#[inline(never)]
#[unsafe(no_mangle)]
fn has_zero_index(xs: &[u8]) -> bool {
    for i in 0..xs.len() {
        if xs[i] == 0 {
            return true;
        }
    }
    false
}

// CHECK-LABEL: foo_index
// CHECK: {{(movq|callq)}} {{\*?}}has_zero_index
// CHECK-NOT: {{(movq|callq)}} {{\*?}}has_zero_index
#[unsafe(no_mangle)]
fn foo_index(xs: &[u8]) {
    println!("a0: {}", has_zero_index(xs));
    println!("b0: {}", has_zero_index(xs));
}

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

// CHECK-LABEL: foo_for
// CHECK: {{(movq|callq)}} {{\*?}}has_zero_for
// CHECK-NOT: {{(movq|callq)}} {{\*?}}has_zero_for
#[unsafe(no_mangle)]
fn foo_for(xs: &[u8]) {
    println!("a1: {}", has_zero_for(xs));
    println!("b1: {}", has_zero_for(xs));
}

#[inline(never)]
#[unsafe(no_mangle)]
fn has_zero_memchr(xs: &[u8]) -> bool {
    xs.contains(&0)
}

// CHECK-LABEL: foo_memchr
// CHECK: {{(movq|callq)}} {{\*?}}has_zero_memchr
// CHECK-NOT: {{(movq|callq)}} {{\*?}}has_zero_memchr
#[unsafe(no_mangle)]
fn foo_memchr(xs: &[u8]) {
    println!("a2: {}", has_zero_memchr(xs));
    println!("b2: {}", has_zero_memchr(xs));
}

#[inline(never)]
#[unsafe(no_mangle)]
fn has_zero_iter(xs: &[u8]) -> bool {
    xs.iter().any(|&x| x == 0)
}

// CHECK-LABEL: foo_iter
// CHECK: {{(movq|callq)}} {{\*?}}has_zero_iter
// CHECK-NOT: {{(movq|callq)}} {{\*?}}has_zero_iter
#[unsafe(no_mangle)]
fn foo_iter(xs: &[u8]) {
    println!("a3: {}", has_zero_iter(xs));
    println!("b3: {}", has_zero_iter(xs));
}

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

// CHECK-LABEL: foo_ptr
// CHECK: {{(movq|callq)}} {{\*?}}has_zero_ptr
// CHECK-NOT: {{(movq|callq)}} {{\*?}}has_zero_ptr
#[unsafe(no_mangle)]
fn foo_ptr(xs: &[u8]) {
    println!("a4: {}", has_zero_ptr(xs));
    println!("b4: {}", has_zero_ptr(xs));
}
