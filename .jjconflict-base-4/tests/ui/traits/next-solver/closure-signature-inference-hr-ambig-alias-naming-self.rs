//@ check-pass
//@ revisions: current next
//@[next] compile-flags: -Znext-solver

// When type checking a closure expr we look at the list of unsolved goals
// to determine if there are any bounds on the closure type to infer a signature from.
//
// We attempt to discard goals that name the closure type so as to avoid inferring the
// closure type to something like `?x = closure(sig=fn(?x))`. This test checks that when
// such a goal names the closure type inside of an ambiguous alias and there exists another
// potential goal to infer the closure signature from, we do that.

trait Trait<'a> {
    type Assoc;
}

impl<'a, F> Trait<'a> for F {
    type Assoc = u32;
}

fn closure_typer1<F>(_: F)
where
    F: Fn(u32) + for<'a> Fn(<F as Trait<'a>>::Assoc),
{
}

fn closure_typer2<F>(_: F)
where
    F: for<'a> Fn(<F as Trait<'a>>::Assoc) + Fn(u32),
{
}

fn main() {
    // Here we have some closure with a yet to be inferred type of `?c`. There are two goals
    // involving `?c` that can be used to determine the closure signature:
    // - `?c: for<'a> Fn<(<?c as Trait<'a>>::Assoc,), Output = ()>`
    // - `?c: Fn<(u32,), Output = ()>`
    //
    // If we were to infer the argument of the closure (`x` below) to `<?c as Trait<'a>>::Assoc`
    // then we would not be able to call `x.into()` as `x` is some unknown type. Instead we must
    // use the `?c: Fn(u32)` goal to infer a signature in order for this code to compile.
    //
    // As the algorithm for picking a goal to infer the signature from is dependent on the ordering
    // of pending goals in the type checker, we test both orderings of bounds to ensure we aren't
    // testing that we just *happen* to pick `?c: Fn(u32)`.
    closure_typer1(move |x| {
        let _: u32 = x.into();
    });
    closure_typer2(move |x| {
        let _: u32 = x.into();
    });
}
