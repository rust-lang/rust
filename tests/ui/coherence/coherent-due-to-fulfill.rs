//@ check-pass
trait Mirror {
    type Assoc;
}
impl<T> Mirror for T {
    type Assoc = T;
}

trait Foo {}
trait Bar {}

// self type starts out as `?0` but is constrained to `()`
// due to the where clause below. Because `(): Bar` does not
// hold in intercrate mode, we can prove the impls disjoint.
impl<T> Foo for T where (): Mirror<Assoc = T> {}
impl<T> Foo for T where T: Bar {}

fn main() {}
