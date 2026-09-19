//@ compile-flags: -Znext-solver=globally

pub trait Family {
    type View<'a>;
}

pub trait Identity {
    type Output: Family;
}

pub trait Carrier {
    type Assoc: Identity<Output = Self::Assoc>;
}

pub fn identity<'a, C: Carrier<Assoc = T>, T: Family>(
    value: T::View<'a>,
) -> <<T as Identity>::Output as Family>::View<'a> {
    value
}
