//@ check-pass

fn main() {}

#[cfg(FALSE)]
fn f1_1(x: isize, ...) {}

#[cfg(FALSE)]
fn f1_2(...) {}

#[cfg(FALSE)]
extern "C" fn f2_1(x: isize, ...) {}

#[cfg(FALSE)]
extern "C" fn f2_2(...) {}

#[cfg(FALSE)]
extern "C" fn f2_3(..., x: isize) {}

#[cfg(FALSE)]
extern fn f3_1(x: isize, ...) {}

#[cfg(FALSE)]
extern fn f3_2(...) {}

#[cfg(FALSE)]
extern fn f3_3(..., x: isize) {}

#[cfg(FALSE)]
extern {
    fn e_f1(...);
    fn e_f2(..., x: isize);
}

struct X;

#[cfg(FALSE)]
impl X {
    fn i_f1(x: isize, ...) {}
    fn i_f2(...) {}
    fn i_f3(..., x: isize, ...) {}
    fn i_f4(..., x: isize, ...) {}
}

#[cfg(FALSE)]
trait T {
    fn t_f1(x: isize, ...) {}
    fn t_f2(x: isize, ...);
    fn t_f3(...) {}
    fn t_f4(...);
    fn t_f5(..., x: isize) {}
    fn t_f6(..., x: isize);
}
