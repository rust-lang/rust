// https://github.com/rust-lang/rust/issues/57924
pub struct Gcm<E>(E);

impl<E> Gcm<E> {
    pub fn crash(e: E) -> Self {
        Self::<E>(e)
        //~^ ERROR type arguments are not allowed on self constructor
    }
}

fn main() {}
