//! Test for #39872 and #39553

fn will_ice(something: &u32) -> impl Iterator<Item = &u32> {
    //~^ ERROR `()` is not an iterator
}

fn main() {}
