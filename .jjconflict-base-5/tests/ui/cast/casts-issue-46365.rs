struct Lorem {
    ipsum: Ipsum //~ ERROR cannot find type `Ipsum`
}

fn main() {
    // Testing `as` casts, so deliberately not using `ptr::null`.
    let _foo: *mut Lorem = 0 as *mut _; // no error here
}
