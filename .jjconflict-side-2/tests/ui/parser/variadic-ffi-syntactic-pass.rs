//@ check-pass

fn main() {}

#[cfg(false)]
fn f1_1(x: isize, _: ...) {}

#[cfg(false)]
fn f1_2(_: ...) {}

#[cfg(false)]
extern "C" fn f2_1(x: isize, _: ...) {}

#[cfg(false)]
extern "C" fn f2_2(_: ...) {}

#[cfg(false)]
extern "C" fn f2_3(_: ..., x: isize) {}

#[cfg(false)]
extern fn f3_1(x: isize, _: ...) {}

#[cfg(false)]
extern fn f3_2(_: ...) {}

#[cfg(false)]
extern fn f3_3(_: ..., x: isize) {}

#[cfg(false)]
extern {
    fn e_f1(...);
    fn e_f2(..., x: isize);
}

struct X;

#[cfg(false)]
impl X {
    fn i_f1(x: isize, _: ...) {}
    fn i_f2(_: ...) {}
    fn i_f3(_: ..., x: isize, _: ...) {}
    fn i_f4(_: ..., x: isize, _: ...) {}
}

#[cfg(false)]
trait T {
    fn t_f1(x: isize, _: ...) {}
    fn t_f2(x: isize, _: ...);
    fn t_f3(_: ...) {}
    fn t_f4(_: ...);
    fn t_f5(_: ..., x: isize) {}
    fn t_f6(_: ..., x: isize);
}
