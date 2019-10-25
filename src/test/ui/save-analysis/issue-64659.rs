// check-pass
// compile-flags: -Zsave-analysis

trait Trait { type Assoc; }

fn main() {
    struct Data<T: Trait> {
        x: T::Assoc,
    }
}
