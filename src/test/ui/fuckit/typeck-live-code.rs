// compile-flags: -Z borrowck-unreachable=no
fn live_code(s: &str) -> bool {
    s //~ ERROR mismatched types
}

fn main() {
    println!("{}", live_code("he he he"));
}
