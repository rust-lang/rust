// check-pass

#[deny(unused_variables)]
fn foo(x: i32, y: i32) -> i32 {
    let z = x + y;
    todo!()
}

#[deny(unused_variables)]
fn bar(x: i32, y: i32) -> i32 {
    todo!("Some message")
}

fn main() {
    foo(0, 1);
    bar(0, 1);
}
