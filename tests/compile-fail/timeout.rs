//error-pattern: reached the configured maximum execution time
#![feature(custom_attribute, attr_literals)]
#![miri(step_limit=1000)]

fn main() {
    for i in 0..1000000 {
        assert!(i < 1000);
    }
}
