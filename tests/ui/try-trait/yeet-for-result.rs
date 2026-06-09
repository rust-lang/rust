//@ run-pass

#![feature(yeet_expr)]

fn always_yeet() -> Result<i32, String> {
    do yeet "hello";
}

fn main() {
    assert_eq!(always_yeet(), Err("hello".to_string()));
}
