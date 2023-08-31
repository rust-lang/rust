struct S {
    x: u8,
}
fn main() {
    let _ = S {
<<<<<<< HEAD //~ ERROR encountered git conflict marker
        x: 42,
=======
        x: 0,
>>>>>>> branch
    }
}
