//@ check-pass

trait I { fn i(&self) -> Self; }

trait A<T:I> {
    fn id(x:T) -> T { x.i() }
}

trait J<T> { fn j(&self) -> T; }

trait B<T:J<T>> {
    fn id(x:T) -> T { x.j() }
}

trait C {
    fn id<T:J<T>>(x:T) -> T { x.j() }
}

pub fn main() { }
