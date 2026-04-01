fn cast_thin_to_fat(x: *const ()) {
    x as *const [u8];
    //~^ ERROR: cannot cast thin pointer `*const ()` to wide pointer `*const [u8]`
}

fn main() {}
