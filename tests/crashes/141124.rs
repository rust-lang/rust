//@ known-bug: #141124
struct S;
trait SimpleTrait {}
trait TraitAssoc {
    type Assoc;
}

impl<T> TraitAssoc for T
where
    T: SimpleTrait,
{
    type Assoc = <(T,) as TraitAssoc>::Assoc;
}
impl SimpleTrait for <S as TraitAssoc>::Assoc {}

pub fn main() {}
