// This test should pass since 'identity' is const fn.

// build-pass (FIXME(62277): could be check-pass?)

fn main() {
    const _FOO: u8 = ::std::convert::identity(42u8);
}
