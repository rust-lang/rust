// check-pass

#[deny(unused_variables)]
fn foo(x: i32, y: i32) -> i32 {
    let z = x + y;
    todo!()
}

fn main() {
    foo(0, 1);
}
