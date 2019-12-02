// check-pass

fn main() {}

#[cfg(FALSE)]
fn f1(x: isize, ...) {}

#[cfg(FALSE)]
extern "C" fn f2(x: isize, ...) {}

#[cfg(FALSE)]
extern fn f3(x: isize, ...) {}

struct X;

#[cfg(FALSE)]
impl X {
    fn f4(x: isize, ...) {}
}

#[cfg(FALSE)]
trait T {
    fn f5(x: isize, ...) {}
    fn f6(x: isize, ...);
}
