//@ revisions: next current
//@[next] compile-flags: -Znext-solver

// cc #119820 the behavior is inconsistent as we discard the where-bound
// candidate for trait goals due to the leak check, but did
// not do so for projection candidates and during normalization.
//
// This results in an inconsistency between `Trait` and `Projection` goals as
// normalizing always constraints the normalized-to term.
trait Trait<'a> {
    type Assoc;
}
impl<'a, T> Trait<'a> for T {
    type Assoc = usize;
}

fn trait_bound<T: for<'a> Trait<'a>>() {}
fn projection_bound<T: for<'a> Trait<'a, Assoc = usize>>() {}

// We use a function with a trivial where-bound which is more
// restrictive than the impl.
fn function1<T: Trait<'static>>() {
    // err
    //
    // Proving `for<'a> T: Trait<'a>` using the where-bound does not
    // result in a leak check failure even though it does not apply.
    // We prefer env candidates over impl candidatescausing this to succeed.
    trait_bound::<T>();
    //[next]~^ ERROR the trait bound `for<'a> T: Trait<'a>` is not satisfied
}

fn function2<T: Trait<'static, Assoc = usize>>() {
    // err
    //
    // Proving the `Projection` goal `for<'a> T: Trait<'a, Assoc = usize>`
    // does not use the leak check when trying the where-bound, causing us
    // to prefer it over the impl, resulting in a placeholder error.
    projection_bound::<T>();
    //[next]~^ ERROR the trait bound `for<'a> T: Trait<'a>` is not satisfied
    //[current]~^^ ERROR mismatched types
}

fn function3<T: Trait<'static, Assoc = usize>>() {
    // err
    //
    // Trying to normalize the type `for<'a> fn(<T as Trait<'a>>::Assoc)`
    // only gets to `<T as Trait<'a>>::Assoc` once `'a` has been already
    // instantiated, causing us to prefer the where-bound over the impl
    // resulting in a placeholder error. Even if we were to also use the
    // leak check during candidate selection for normalization, this
    // case would still not compile.
    let _higher_ranked_norm: for<'a> fn(<T as Trait<'a>>::Assoc) = |_| ();
    //[next]~^ ERROR higher-ranked subtype error
    //[next]~| ERROR higher-ranked subtype error
    //[current]~^^^ ERROR mismatched types
    //[current]~| ERROR mismatched types
}

fn main() {}
