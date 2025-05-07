//@ run-rustfix
#![allow(dead_code)]

trait GatTrait {
    type Gat<'a> where Self: 'a;

    fn test(&self) -> Self::Gat<'_>;
}

trait SuperTrait<T>
where
    Self: 'static,
    for<'a> Self: GatTrait<Gat<'a> = &'a T>,
{
    fn copy(&self) -> Self::Gat<'_> where T: Copy {
        *self.test()
        //~^ ERROR mismatched types
    }
}

fn main() {}
