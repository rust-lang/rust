// Verify that debuginfo column numbers are 1-based byte offsets.
//
//@ ignore-msvc
//@ compile-flags: -C debuginfo=2

#[rustfmt::skip]
fn main() {
    unsafe {
        // Column numbers are 1-based. Regression test for #65437.
        // CHECK: call void @giraffe(){{( #[0-9]+)?}}, !dbg [[A:!.*]]
        giraffe();

        // Column numbers use byte offests. Regression test for #67360
        // CHECK: call void @turtle(){{( #[0-9]+)?}}, !dbg [[B:!.*]]
/* ż */ turtle();

        // CHECK: [[A]] = !DILocation(line: 11, column: 9,
        // CHECK: [[B]] = !DILocation(line: 15, column: 10,
    }
}

extern "C" {
    fn giraffe();
    fn turtle();
}
