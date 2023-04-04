#[link(name = "library")]
extern "C" {
    fn overflow();
}

fn main() {
    unsafe {
        overflow();
    }
}
