// compile-flags: -Z chalk

trait Eq { }
trait Hash: Eq { }

impl Eq for i32 { }
impl Hash for i32 { }

struct Set<T: Hash> {
    _x: T,
}

fn only_eq<T: Eq>() { }

fn take_a_set<T>(_: &Set<T>) {
    // `Set<T>` is an input type of `take_a_set`, hence we know that
    // `T` must implement `Hash`, and we know in turn that `T` must
    // implement `Eq`.
    only_eq::<T>()
}

fn main() {
    let set = Set {
        _x: 5,
    };

    take_a_set(&set);
}
