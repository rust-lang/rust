#[link(name = "library", kind = "static")]
extern "C" {
    fn overflow();
}

fn main() {
    unsafe {
        overflow();
    }
}
