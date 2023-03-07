extern "C" {
    fn overflow();
}

fn main() {
    unsafe { overflow() }
}
