// run-pass
// issue #89688

#![feature(let_else)]

fn example_let_else(value: Option<String>) {
    let Some(inner) = value else {
        println!("other: {:?}", value); // OK
        return;
    };
    println!("inner: {}", inner);
}

fn main() {
    example_let_else(Some("foo".into()));
    example_let_else(None);
}
