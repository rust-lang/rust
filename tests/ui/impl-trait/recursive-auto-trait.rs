//@ check-pass
fn is_send<T: Send>(_: T) {}
fn foo() -> impl Send {
    if false {
        is_send(foo());
    }
    ()
}

fn main() {}
