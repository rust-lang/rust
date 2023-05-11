// check-fail
// compile-flags: -Z tiny-const-eval-limit

const fn recurse(n: u32) -> u32 {
    if n == 0 {
        n
    } else {
        recurse(n - 1) //~ ERROR evaluation of constant value failed [E0080]
    }
}

const X: u32 = recurse(19);

fn main() {
    println!("{X}");
}
