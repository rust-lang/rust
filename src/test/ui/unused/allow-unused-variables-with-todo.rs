// check-pass

#[deny(unused_variables)]
fn plain(x: i32, y: i32) -> i32 {
    todo!()
}

#[deny(unused_variables)]
fn message(x: i32, y: i32) -> i32 {
    todo!("message")
}

#[deny(unused_variables)]
fn statement(x: i32, y: i32) -> i32 {
    let z = x + y;
    todo!()
}

#[deny(unused_variables)]
fn statement_message(x: i32, y: i32) -> i32 {
    let z = x + y;
    todo!("message")
}

fn main() {
    plain(0, 1);
    message(0, 1);
    statement(0, 1);
    statement_message(0, 1);
}
