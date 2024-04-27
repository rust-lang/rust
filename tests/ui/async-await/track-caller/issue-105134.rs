//@ check-pass
//@ edition:2021

#[track_caller]
fn f() {
    let _ = async {};
}

fn main() {
    f();
}
