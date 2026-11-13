struct S {
    x: u8,
}
fn main() {
    let _ = S {
<<<<<<< HEAD //~ ERROR encountered diff marker
        x: 42,
=======
        x: 0,
>>>>>>> branch
    }
}
