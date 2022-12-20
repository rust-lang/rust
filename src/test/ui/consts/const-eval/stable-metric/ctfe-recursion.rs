// check-pass

const fn recurse(n: u32) -> u32 {
    if n == 0 {
        n
    } else {
        recurse(n - 1)
    }
}

const X: u32 = recurse(30);

fn main() {
    println!("{X}");
}
