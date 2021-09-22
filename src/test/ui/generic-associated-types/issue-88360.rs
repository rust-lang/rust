#![feature(generic_associated_types)]

trait GatTrait {
    type Gat<'a>;

    fn test(&self) -> Self::Gat<'_>;
}

trait SuperTrait<T>
where
    for<'a> Self: GatTrait<Gat<'a> = &'a T>,
{
    fn copy(&self) -> Self::Gat<'_> where T: Copy {
        *self.test()
        //~^ mismatched types
    }
}

fn main() {}
