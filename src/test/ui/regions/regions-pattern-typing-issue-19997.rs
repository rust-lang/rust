// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

fn main() {
    let a0 = 0;
    let f = 1;
    let mut a1 = &a0;
    match (&a1,) {
        (&ref b0,) => {
            a1 = &f; //[ast]~ ERROR cannot assign
            //[mir]~^ ERROR cannot assign to `a1` because it is borrowed
            drop(b0);
        }
    }
}
