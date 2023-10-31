extern crate no_builtins;

#[no_mangle]
fn call_foo() {
    no_builtins::foo();
}

fn main() {
    call_foo();
}
