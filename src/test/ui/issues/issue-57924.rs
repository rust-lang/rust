pub struct Gcm<E>(E);

impl<E> Gcm<E> {
    pub fn crash(e: E) -> Self {
        Self::<E>(e)
        //~^ ERROR type arguments are not allowed for this type
    }
}

fn main() {}
