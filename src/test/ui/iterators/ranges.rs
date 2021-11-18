fn main() {
    for _ in ..10 {}
    //~^ ERROR E0277
    for _ in ..=10 {}
    //~^ ERROR E0277
    for _ in 0..10 {}
    for _ in 0..=10 {}
    for _ in 0.. {}
}

fn references_do_not_coerce_to_ptr_range(start: &i32, end: &i32) {
    // Better not turn into Range<*const i32>.
    for _ in start..end {}
    //~^ ERROR E0277
}
