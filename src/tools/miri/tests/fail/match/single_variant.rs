// Ideally, this would be UB regardless of #[non_exhaustive]. For now,
// at least the semantics don't depend on the crate you're in.
//
// See: rust-lang/rust#147722
#![allow(dead_code)]

#[repr(u8)]
enum Exhaustive {
    A(u8) = 42,
}

#[repr(u8)]
#[non_exhaustive]
enum NonExhaustive {
    A(u8) = 42,
}

fn main() {
    unsafe {
        let x: &[u8; 2] = &[21, 37];
        let y: &Exhaustive = std::mem::transmute(x);
        match y {
            Exhaustive::A(_) => {},
        }

        let y: &NonExhaustive = std::mem::transmute(x);
        match y { //~ ERROR: enum value has invalid tag
            NonExhaustive::A(_) => {},
        }
    }
}
