// This test should pass since 'identity' is const fn.

//@ check-pass

fn main() {
    const _FOO: u8 = ::std::convert::identity(42u8);
}
