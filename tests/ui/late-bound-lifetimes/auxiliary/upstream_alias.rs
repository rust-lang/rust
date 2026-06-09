pub trait Trait<'a> {
    type Assoc;
}

pub type Alias<'a, T> = <T as Trait<'a>>::Assoc;
