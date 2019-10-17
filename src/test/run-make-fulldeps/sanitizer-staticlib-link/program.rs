#[link(name = "library")]
extern {
    fn overflow();
}

fn main() {
    unsafe {
        overflow();
    }
}
