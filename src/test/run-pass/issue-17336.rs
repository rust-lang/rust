#[allow(dead_code)]
fn check(a: &str) {
    let x = a as *const str;
    x == x;
}

fn main() {}
