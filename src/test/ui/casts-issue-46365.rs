struct Lorem {
    ipsum: Ipsum //~ ERROR cannot find type `Ipsum`
}

fn main() {
    let _foo: *mut Lorem = core::ptr::null_mut(); // no error here
}
