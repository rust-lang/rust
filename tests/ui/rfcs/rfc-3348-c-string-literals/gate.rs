// gate-test-c_str_literals

macro_rules! m {
    ($t:tt) => {}
}

fn main() {
    c"foo";
    //~^ ERROR: `c".."` literals are experimental

    m!(c"test");
    //~^ ERROR: `c".."` literals are experimental
}
