pub struct Gcm<E>(E);

impl<E> Gcm<E> {
    pub fn crash(e: E) -> Self {
        Self::<E>(e)
    }
}

fn main() {}
