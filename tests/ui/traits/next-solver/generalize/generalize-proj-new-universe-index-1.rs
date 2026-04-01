//@ compile-flags: -Znext-solver
//@ check-pass

// A minimization of an ambiguity when using typenum. See
// https://github.com/rust-lang/trait-system-refactor-initiative/issues/55
// for more details.
trait Id {
    type Assoc: ?Sized;
}
impl<T: ?Sized> Id for T {
    type Assoc = T;
}

trait WithAssoc<T: ?Sized> {
    type Assoc: ?Sized;
}


struct Leaf;
struct Wrapper<U: ?Sized>(U);

impl<U: ?Sized> WithAssoc<U> for Leaf {
    type Assoc = U;
}

impl<Ul: ?Sized, Ur: ?Sized> WithAssoc<Wrapper<Ur>> for Wrapper<Ul>
where
    Ul: WithAssoc<Ur>,
{
    type Assoc = <<Ul as WithAssoc<Ur>>::Assoc as Id>::Assoc;
}

fn bound<T: ?Sized, U: ?Sized, V: ?Sized>()
where
    T: WithAssoc<U, Assoc = V>,
{
}

// normalize self type to `Wrapper<Leaf>`
//   This succeeds, HOWEVER, instantiating the query response previously
//   incremented the universe index counter.
// equate impl headers:
//      <Wrapper<Leaf> as WithAssoc<<Wrapper<Leaf> as Id>::Assoc>>
//      <Wrapper<?2t> as WithAssoc<Wrapper<?3t>>>
// ~> AliasRelate(<Wrapper<Leaf> as Id>::Assoc, Equate, Wrapper<?3t>)
// add where bounds:
// ~> Leaf: WithAssoc<?3t>
// equate with assoc type:
//      ?0t
//      <Leaf as WithAssoc<?3t>>::Assoc as Id>::Assoc
// ~> AliasRelate(
//      <<Leaf as WithAssoc<?3t>>::Assoc as Id>::Assoc,
//      Equate,
//      <<Leaf as WithAssoc<?4t>>::Assoc as Id>::Assoc,
//    )
//
// We do not reuse `?3t` during generalization because `?0t` cannot name `?4t` as we created
// it after incrementing the universe index while normalizing the self type.
//
// evaluate_added_goals_and_make_query_response:
//    AliasRelate(<Wrapper<Leaf> as Id>::Assoc, Equate, Wrapper<?3t>)
//      YES, constrains ?3t to Leaf
//    AliasRelate(
//      <<Leaf as WithAssoc<Leaf>>::Assoc as Id>::Assoc,
//      Equate,
//      <<Leaf as WithAssoc<?4t>>::Assoc as Id>::Assoc,
//    )
//
// Normalizing <<Leaf as WithAssoc<?4t>>::Assoc as Id>::Assoc then *correctly*
// results in ambiguity.
fn main() {
    bound::<<Wrapper<Leaf> as Id>::Assoc, <Wrapper<Leaf> as Id>::Assoc, _>()
}
