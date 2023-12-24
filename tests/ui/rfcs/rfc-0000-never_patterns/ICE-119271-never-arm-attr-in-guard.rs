fn main() {}

fn attr_in_guard() {
    match None::<u32> {
        Some(!) //~ ERROR `!` patterns are experimental
            if #[deny(unused_mut)] //~ ERROR attributes on expressions are experimental
            false //~ ERROR a guard on a never pattern will never be run
    }
    match false {}
}
