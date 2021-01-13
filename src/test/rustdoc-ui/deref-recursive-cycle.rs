// check-pass
// #26207: Ensure `Deref` cycles are properly handled without errors.

#[derive(Copy, Clone)]
struct S;

impl std::ops::Deref for S {
    type Target = S;

    fn deref(&self) -> &S {
        self
    }
}

fn main() {
    let s: S = *******S;
}
