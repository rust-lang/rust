// compile-flags: -D warnings
// check-pass

fn main() {
    #[warn(unused)]
    let a = 5;
    //~^ WARNING: unused
}
