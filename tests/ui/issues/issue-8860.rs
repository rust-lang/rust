//@ run-pass
#![allow(dead_code)]

static mut DROP: isize = 0;
static mut DROP_S: isize = 0;
static mut DROP_T: isize = 0;

struct S;
impl Drop for S {
    fn drop(&mut self) {
        unsafe {
            DROP_S += 1;
            DROP += 1;
        }
    }
}
fn f(ref _s: S) {}

struct T { i: isize }
impl Drop for T {
    fn drop(&mut self) {
        unsafe {
            DROP_T += 1;
            DROP += 1;
        }
    }
}
fn g(ref _t: T) {}

fn do_test() {
    let s = S;
    f(s);
    unsafe {
        assert_eq!(1, DROP);
        //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
        assert_eq!(1, DROP_S);
        //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
    }
    let t = T { i: 1 };
    g(t);
    unsafe { assert_eq!(1, DROP_T); }
    //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
}

fn main() {
    do_test();
    unsafe {
        assert_eq!(2, DROP);
        //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
        assert_eq!(1, DROP_S);
        //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
        assert_eq!(1, DROP_T);
        //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
    }
}
