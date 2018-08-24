// error-pattern:explicit panic

pub fn main() {
    panic!();
    println!("{}", 1);
}
