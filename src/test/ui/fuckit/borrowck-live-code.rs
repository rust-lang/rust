// compile-flags: -Z borrowck-unreachable=no
// build-fail
fn live_code(s: &str) -> &'static str {
    s //~ ERROR lifetime may not live long enough
}

fn main() {
    println!("{}", live_code("he he he"));
}
