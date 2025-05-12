// Check that draining at the front or back doesn't copy memory.

//@ compile-flags: -Copt-level=3
//@ needs-deterministic-layouts
//@ ignore-std-debug-assertions (FIXME: checks for call detect scoped noalias metadata)

#![crate_type = "lib"]

use std::collections::VecDeque;

// CHECK-LABEL: @clear
// CHECK-NOT: call
// CHECK-NOT: br
// CHECK: getelementptr inbounds
// CHECK-NEXT: {{call void @llvm.memset|store}}
// CHECK-NEXT: ret void
#[no_mangle]
pub fn clear(v: &mut VecDeque<i32>) {
    v.drain(..);
}

// CHECK-LABEL: @truncate
// CHECK-NOT: call
// CHECK: br
// CHECK-NOT: call
// CHECK: br
// CHECK-NOT: call
// CHECK: br
// CHECK-NOT: call
// CHECK: br
// CHECK-NOT: call
// CHECK-NOT: br
// CHECK: ret void
#[no_mangle]
pub fn truncate(v: &mut VecDeque<i32>, n: usize) {
    if n < v.len() {
        v.drain(n..);
    }
}

// CHECK-LABEL: @advance
// CHECK-NOT: call
// CHECK: br
// CHECK-NOT: call
// CHECK: br
// CHECK-NOT: call
// CHECK: br
// CHECK-NOT: call
// CHECK: br
// CHECK-NOT: call
// CHECK: br
// CHECK-NOT: call
// CHECK-NOT: br
// CHECK: ret void
#[no_mangle]
pub fn advance(v: &mut VecDeque<i32>, n: usize) {
    if n < v.len() {
        v.drain(..n);
    } else {
        v.clear();
    }
}

// CHECK-LABEL: @remove
// CHECK: call
// CHECK: ret void
#[no_mangle]
pub fn remove(v: &mut VecDeque<i32>, a: usize, b: usize) {
    v.drain(a..b);
}
