// rustfmt-format_strings: true

macro_rules! test {
    () => {
        fn from() {
            None.expect(
                "We asserted that `buffer.len()` is exactly `$n` so we can expect `ApInt::from_iter` to be successful.",
            )
        }
    };
}
