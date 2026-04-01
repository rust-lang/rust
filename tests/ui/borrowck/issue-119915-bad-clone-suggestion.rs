use std::marker::PhantomData;

struct Example<E, FakeParam>(PhantomData<(fn(E), fn(FakeParam))>);

struct NoLifetime;
struct Immutable<'a>(PhantomData<&'a ()>);

impl<'a, E: 'a> Copy for Example<E, Immutable<'a>> {}
impl<'a, E: 'a> Clone for Example<E, Immutable<'a>> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E, FakeParam> Example<E, FakeParam> {
    unsafe fn change<NewFakeParam>(self) -> Example<E, NewFakeParam> {
        Example(PhantomData)
    }
}

impl<E> Example<E, NoLifetime> {
    fn the_ice(&mut self) -> Example<E, Immutable<'_>> {
        unsafe { self.change() }
        //~^ ERROR cannot move out of `*self` which is behind a mutable reference
    }
}

fn main() {}
