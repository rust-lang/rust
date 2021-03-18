// Verify that debuginfo column nubmers are 1-based byte offsets.
//
// ignore-windows
// compile-flags: -C debuginfo=2

fn main() {
    unsafe {
        // Column numbers are 1-based. Regression test for #65437.
        // CHECK: call void @giraffe(), !dbg [[A:!.*]]
        giraffe();

        // Column numbers use byte offests. Regression test for #67360
        // CHECK: call void @turtle(), !dbg [[B:!.*]]
/* ż */ turtle();

        // CHECK: [[A]] = !DILocation(line: 10, column: 9,
        // CHECK: [[B]] = !DILocation(line: 14, column: 10,
    }
}

extern "C" {
    fn giraffe();
    fn turtle();
}
