struct Lorem {
    ipsum: Ipsum //~ ERROR cannot find type `Ipsum`
}

fn main() {
    let _foo: *mut Lorem = core::ptr::NonNull::dangling().as_ptr(); // no error here
}
