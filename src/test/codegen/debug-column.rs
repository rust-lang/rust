// Verify that emitted debuginfo column nubmers are 1-based. Regression test for issue #65437.
//
// ignore-windows
// compile-flags: -C debuginfo=2

fn main() {
    unsafe {
        // CHECK: call void @giraffe(), !dbg [[DBG:!.*]]
        // CHECK: [[DBG]] = !DILocation(line: 10, column: 9
        giraffe();
    }
}

extern {
    fn giraffe();
}
