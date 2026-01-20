//~ ERROR: cycle detected when getting HIR ID of `Default`
trait Default {
    type Id;

    fn intu(&self) -> &Self::Id;
}

impl<T: Default<Id = U>, U: Copy> Default for U {
    default type Id = T;
    fn intu(&self) -> &Self::Id {
        self
    }
}

fn specialization<T>(t: T) -> U {
    *t.intu()
}

use std::num::NonZero;

fn main() {
    let assert_eq = NonZero::<u8, Option<NonZero<u8>>>(0);
    assert_eq!(specialization, None);
}
