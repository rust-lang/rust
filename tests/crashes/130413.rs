//@ known-bug: #130413

#![feature(transmutability)]
trait Aaa {
    type Y;
}

trait Bbb {
    type B: std::mem::TransmuteFrom<()>;
}

impl<T> Bbb for T
where
    T: Aaa,
{
    type B = T::Y;
}
