//@ check-pass
//@ compile-flags: -Znext-solver

fn unconstrained<T>() -> T {
    todo!()
}

trait Relate<T> {}

struct Type<T>(T);

impl<T> Relate<Type<T>> for T {}

fn relate_types<T: Relate<U>, U>(_: &T, _: &U) {}

fn type_inference_var() {
    let low_universe = unconstrained();

    let bump: for<'a, 'b> fn(&'a (), &'b ()) = |_, _| ();
    let _: for<'a> fn(&'a (), &'a ()) = bump;

    let high_universe = unconstrained();
    relate_types(&high_universe, &low_universe);

    let _: () = high_universe;
    let _: Type<()> = low_universe;
}

#[derive(Copy, Clone)]
struct Const<const N: usize>;

trait RelateConst<const N: usize> {}

impl<const N: usize> RelateConst<{ N }> for Const<N> {}

fn relate_consts<const N: usize, const M: usize>(_: Const<N>, _: Const<M>)
where
    Const<N>: RelateConst<M>,
{
}

fn const_inference_var() {
    let low_universe = Const::<_>;

    let bump: for<'a, 'b> fn(&'a (), &'b ()) = |_, _| ();
    let _: for<'a> fn(&'a (), &'a ()) = bump;

    let high_universe = Const::<_>;
    relate_consts(high_universe, low_universe);

    let _: Const<0> = high_universe;
    let _: Const<0> = low_universe;
}

fn main() {
    type_inference_var();
    const_inference_var();
}
