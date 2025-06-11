// Ensure that we properly deal with missing signatures and inferred types `_` in the signature.
#![feature(generic_const_items)]
#![expect(incomplete_features)]

// For implicit & explicit inferred types in the item signature we generally try to infer the
// type of the body which we can then suggest to the user.

// However, in the type of trait impl assoc items specifically we first try to suggest the type of
// the corresponding definition in the trait. While doing so, we once didn't use to instantiate GACs
// correctly & couldn't handle mismatches in parameter list lengths very well (impl v trait).
// issue: <https://github.com/rust-lang/rust/issues/124833>

trait Trait {
    const K<T>: T;
    const Q<'a>: &'a str;
}

impl Trait for () {
    const K<T> = ();
    //~^ ERROR missing type for `const` item
    //~| ERROR mismatched types
    //~| SUGGESTION  ()
     const Q = "";
    //~^ ERROR missing type for `const` item
    //~| ERROR lifetime parameters or bounds on associated constant `Q` do not match the trait declaration
    //~| SUGGESTION : &str
}

// For parametrized free const items however, we can't typeck the body without causing a query cycle
// so we don't and thus fall back to a generic suggestion that has a placeholder.

const _<T> = loop {};
//~^ ERROR missing type for `const` item
//~| SUGGESTION <type>

const _<T>: _ = 0;
//~^ ERROR the placeholder `_` is not allowed

fn main() {}
