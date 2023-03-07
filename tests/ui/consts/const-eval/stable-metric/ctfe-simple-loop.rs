// check-fail
// compile-flags: -Z tiny-const-eval-limit
const fn simple_loop(n: u32) -> u32 {
    let mut index = 0;
    while index < n { //~ ERROR evaluation of constant value failed [E0080]
        index = index + 1;
    }
    0
}

const X: u32 = simple_loop(19);

fn main() {
    println!("{X}");
}
