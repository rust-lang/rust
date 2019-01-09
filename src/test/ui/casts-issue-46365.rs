struct Lorem {
    ipsum: Ipsum //~ ERROR cannot find type `Ipsum`
}

fn main() {
    let _foo: *mut Lorem = 0 as *mut _; // no error here
}
