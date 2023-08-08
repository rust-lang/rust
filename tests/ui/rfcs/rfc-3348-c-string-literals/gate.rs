// gate-test-c_str_literals
// known-bug: #113333
// edition: 2021

macro_rules! m {
    ($t:tt) => {}
}

fn main() {
    c"foo";
    // FIXME(c_str_literals): This should be ``c".."` literals are experimental`

    m!(c"test");
    // FIXME(c_str_literals): This should be ``c".."` literals are experimental`
}
