// check-pass
const fn simple_loop(n: u32) -> u32 {
    let mut index = 0;
    let mut res = 0;
    while index < n {
        res = res + index;
        index = index + 1;
    }
    res
}

const X: u32 = simple_loop(30);

fn main() {
    println!("{X}");
}
