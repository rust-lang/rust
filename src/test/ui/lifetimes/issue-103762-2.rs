// Failure cases for #103762.

trait Trait {}

// Case 1: '_ in trait object bound without any other anonymous/elided lifetimes.
// Pre-1.64 rejected this, but 1.64 did not. Make sure it does.
fn a1(_: Box<dyn Trait + '_>) -> &str { loop {} } //~ ERROR missing lifetime specifier
fn a2(_: Box<Trait + '_>) -> &str { loop {} } //~ ERROR missing lifetime specifier
//~^ WARNING deprecated

// Case 2: Inside fn() type.
// Pre-1.64 and 1.64 rejected this, should still reject.
fn b1(_: fn(&(dyn Trait + '_)) -> &str) { loop {} } //~ ERROR missing lifetime specifier
fn b2(_: fn(&(Trait + '_)) -> &str) { loop {} } //~ ERROR missing lifetime specifier
//~^ WARNING deprecated

// Case 3: Inside generic bound.
// Pre-1.64 and 1.64 rejected this, should still reject.
fn c1<T: Trait + '_>(_: &T) -> &str { loop {} } //~ ERROR `'_` cannot be used here
fn c2<T>(_: &T) -> &str where T: Trait + '_ { loop {} } //~ ERROR `'_` cannot be used here

// Case 4.a: Inside impl Trait without any other anonymous/elided lifetimes.
//  * Pre-1.64 rejected this because it thought `+ '_` needed to be a named lifetime,
//    and did not report on `&str`.
//  * 1.64 rejected this because of unstable anonymous_lifetime_in_impl_trait,
//    and because `&str` needs a named lifetime.
// It makes sense to keep doing what 1.64 did.
fn d1(_: impl Trait + '_) //~ ERROR anonymous lifetimes in `impl Trait` are unstable
    -> &str { loop {} } //~ ERROR missing lifetime specifier

// Case 4.b: Inside impl Trait with other anonymous/elided lifetimes.
//  * Pre-1.64 rejected these because it thought `+ '_` needed to be a named lifetime.
//  * 1.64 rejected these because of unstable anonymous_lifetime_in_impl_trait,
//    and because it thought `&str` needed a named lifetime parameter.
// It should continue reporting on the unstable feature, but not on `&str`, since we
// shouldn't count `+ '_` as a new elision candidate.
fn d2(_: &(impl Trait + '_)) //~ ERROR anonymous lifetimes in `impl Trait` are unstable
    -> &str { loop {} }
fn d3(_: &i32, _: impl Trait + '_) //~ ERROR anonymous lifetimes in `impl Trait` are unstable
    -> &str { loop {} }
fn d4(_: impl Trait + '_, _: &i32) //~ ERROR anonymous lifetimes in `impl Trait` are unstable
    -> &str { loop {} }

fn main() {}
