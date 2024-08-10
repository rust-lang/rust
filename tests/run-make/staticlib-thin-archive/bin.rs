#[link(name = "rust_archive", kind = "static")]
extern "C" {
    fn simple_fn();
}

fn main() {
    unsafe {
        simple_fn();
    }
}
