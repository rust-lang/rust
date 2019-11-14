// check-pass

fn foo() -> [u8; 4 * 1024 * 1024 * 1024 * 1024] {
    unimplemented!()
}

fn main() {
    foo();
}
