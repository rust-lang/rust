//! Regression test for #159685

struct EntriesBuffer(Box<[u8]>);

impl EntriesBuffer {
    fn has_lifetime(&self) -> impl Iterator<Item = &mut str> {
        //~^ ERROR expected `IterMut<'_, u8>` to be an iterator that yields `&mut str`, but it yields `&mut u8`
        self.0.iter_mut()
    }
}

fn main() {
    EntriesBuffer(vec![0u8].into_boxed_slice()).has_lifetime();
}
