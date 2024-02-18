//@ check-pass
//@ pretty-expanded FIXME #23616

trait T {
    unsafe extern "Rust" fn foo(&self);
}

impl T for () {
    unsafe extern "Rust" fn foo(&self) {}
}

fn main() {}
