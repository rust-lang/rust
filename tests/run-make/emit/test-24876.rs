// Checks for issue #24876

fn main() {
    let mut v = 0;
    for i in 0..0 {
        v += i;
    }
    println!("{}", v)
}
