use std::ops::Deref;

pub struct Pin<P>(P);

impl<P, T> Deref for Pin<P>
where
    P: Deref<Target=T>,
{
    type Target = T;

    fn deref(&self) -> &T {
        &*self.0
    }
}

impl<P> Pin<P> {
    fn poll(self) {}
}

fn main() {
    let mut unit = ();
    let pin = Pin(&mut unit);
    pin.poll();
}
