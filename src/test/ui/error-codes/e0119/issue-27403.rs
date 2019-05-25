pub struct GenX<S> {
    inner: S,
}

impl<S> Into<S> for GenX<S> { //~ ERROR conflicting implementations
    fn into(self) -> S {
        self.inner
    }
}

fn main() {}
