//@ check-pass

trait T {
    unsafe extern "Rust" fn foo(&self);
}

impl T for () {
    unsafe extern "Rust" fn foo(&self) {}
}

fn main() {}
