// Traits with type associated consts are dyn compatible. However, all associated consts must
// be specified in the corresp. trait object type (barring exceptions) similiar to associated
// types. Check that we reject code that doesn't provide the necessary bindings.

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait Trait {
    #[type_const]
    const K: usize;
}

// fn ctxt / body
fn main() {
    let _: dyn Trait;
    //~^ ERROR the value of the associated constant `K` in `Trait` must be specified
}

// item ctxt / signature / non-body
struct Store(dyn Trait);
//~^ ERROR the value of the associated constant `K` in `Trait` must be specified

// item ctxt & no wfcking (eager ty alias)
type DynTrait = dyn Trait;
//~^ ERROR the value of the associated constant `K` in `Trait` must be specified
