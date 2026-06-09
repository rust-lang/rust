fn main() {
    let a0 = 0;
    let f = 1;
    let mut a1 = &a0;
    match (&a1,) {
        (&ref b0,) => {
            a1 = &f; //~ ERROR cannot assign to `a1` because it is borrowed
            drop(b0);
        }
    }
}
