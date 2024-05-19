//@ known-bug: #123810
//@ compile-flags: -Zlint-mir

fn temp() -> (String, i32) {
    (String::from("Hello"), 1)
}

fn main() {
    let f = if true { &temp() } else { &temp() };
}
