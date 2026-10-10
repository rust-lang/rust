// We currently do not check trait objects for well formedness.
// However, previously we desugared all const arguments to anon
// consts which resulted in us happening to check that they were
// of the correct type. Nowadays we don't necessarily lower to a
// const argument, but to continue erroring on such code we special
// case `ConstArgHasType` clauses to be checked for trait objects
// even though we ignore the rest of the wf requirements.

trait Object<const N: usize> {}
trait Object2<T> {}
trait Object3<'a: 'static, const N: usize> {}

struct Wrap<T>(T);

fn arg<const B: bool, const N: usize>(
    param: &dyn Object<B>,
    //~^ ERROR: the constant `B` is not of type
    anon: &dyn Object<true>,
    //~^ ERROR: mismatched types
) {
}

fn indirect<const B: bool, const N: usize>(
    param: &dyn Object2<Wrap<[(); B]>>,
    //~^ ERROR: the constant `B` is not of type
    anon: &dyn Object2<Wrap<[(); true]>>,
    //~^ ERROR: mismatched types
) {
}

// the 'a: 'static bound should *not* have its error reported (at least, until we implement checking
// all wf things for dyn objects)
fn binder<const B: bool>(param: &dyn for<'a> Object3<'a, B>) {}
//~^ ERROR: the constant `B` is not of type

fn main() {}
