// check-fail
// compile-flags: -Z tiny-const-eval-limit

#[rustfmt::skip]
const fn recurse(n: u32) -> u32 {
    if n == 0 {
        n
    } else {
        recurse(n - 1) //~ ERROR is taking a long time
    }
}

const X: u32 = recurse(19);

fn main() {
    println!("{X}");
}
