// Regression tests for https://github.com/rust-lang/rust/issues/81342
// foo and bar should generate the same, efficient assembly code
//@ compile-flags: -O
//@ only-64bit
#![crate_type = "lib"]

use std::num::NonZeroUsize;

// CHECK-LABEL: @foo
// CHECK-COUNT-2: icmp eq i64 %{{.*}}, 0
// CHECK: or i1
// CHECK: select
// CHECK-NOT: icmp
#[unsafe(no_mangle)]
pub fn foo(x: Option<NonZeroUsize>, y: Option<NonZeroUsize>) -> Option<NonZeroUsize> {
    if let (Some(x2), Some(y2)) = (x, y) { NonZeroUsize::new(x2.get() + y2.get()) } else { None }
}

// CHECK-LABEL: @bar
// CHECK-COUNT-2: icmp eq i64 %{{.*}}, 0
// CHECK: or i1
// CHECK: select
// CHECK-NOT: icmp
#[unsafe(no_mangle)]
pub fn bar(x: Option<NonZeroUsize>, y: Option<NonZeroUsize>) -> Option<NonZeroUsize> {
    x.zip(y).map(|(x2, y2)| NonZeroUsize::new(x2.get() + y2.get())).flatten()
}
