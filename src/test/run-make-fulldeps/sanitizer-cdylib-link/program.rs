extern {
    fn overflow();
}

fn main() {
    unsafe { overflow() }
}
