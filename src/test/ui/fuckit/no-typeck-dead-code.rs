// compile-flags: -Z borrowck-unreachable=no
// run-pass

fn dead_code(s: &str) -> bool { //~ WARNING never used
    true
}

fn main() {
    println!("he he he")
}
