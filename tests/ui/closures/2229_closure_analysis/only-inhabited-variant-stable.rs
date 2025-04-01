// This example used to compile, but the fact that it should was never properly
// discussed. With further experience, we concluded that capture precision
// depending on whether some types are inhabited goes too far, introducing a
// bunch of headaches without much benefit.
//@ edition:2021
enum Void {}

pub fn main() {
    let mut r = Result::<Void, (u32, u32)>::Err((0, 0));
    let mut f = || {
        let Err((ref mut a, _)) = r;
        *a = 1;
    };
    let mut g = || {
    //~^ ERROR: cannot borrow `r` as mutable more than once at a time
        let Err((_, ref mut b)) = r;
        *b = 2;
    };
    f();
    g();
    assert!(matches!(r, Err((1, 2))));
}
