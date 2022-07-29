struct Ooopsies<const N: u8 = { u8::MAX + 1 }>;
//~^ error: evaluation of constant value failed

trait Trait<const N: u8> {}
impl Trait<3> for () {}
struct WhereClause<const N: u8 = 2> where (): Trait<N>;
//~^ error: the trait bound `(): Trait<2_u8>` is not satisfied

trait Traitor<T, const N: u8> {}
struct WhereClauseTooGeneric<T = u32, const N: u8 = 2>(T) where (): Traitor<T, N>;

// no error on struct def
struct DependentDefaultWfness<const N: u8 = 1, T = WhereClause<N>>(T);
fn foo() -> DependentDefaultWfness {
    //~^ error: the trait bound `(): Trait<1_u8>` is not satisfied
    loop {}
}

fn main() {}
