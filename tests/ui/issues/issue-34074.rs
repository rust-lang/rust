// run-pass
// Make sure several unnamed function parameters don't conflict with each other

trait Tr {
    #[allow(anonymous_parameters)]
    fn f(u8, u8) {}
}

fn main() {
}
