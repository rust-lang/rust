#[link(name = "foo", kind = "static")] // linker should drop this library, no symbols used
#[link(name = "bar", kind = "static")] // symbol comes from this library
#[link(name = "foo", kind = "static")] // now linker picks up `foo` b/c `bar` library needs it
extern "C" {
    fn bar();
}

fn main() {
    unsafe { bar() }
}
