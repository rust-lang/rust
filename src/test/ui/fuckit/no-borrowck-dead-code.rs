// compile-flags: -Z borrowck-unreachable=no
// run-pass

fn dead_code(s: &str) -> &'static str {
    s
}

fn main() {
    println!("he he he")
}
