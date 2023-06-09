// run-pass
macro_rules! m { (<$t:ty>) => { stringify!($t) } }
fn main() {
    println!("{}", m!(<Vec<i32>>));
}
