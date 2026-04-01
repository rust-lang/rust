// Check that #[thread_local] attribute is rejected on non-static items.
#![feature(thread_local)]

#[thread_local]
//~^ ERROR `#[thread_local]` attribute cannot be used on constants
const A: u32 = 0;

#[thread_local]
//~^ ERROR `#[thread_local]` attribute cannot be used on functions
fn main() {
    #[thread_local] || {};
    //~^ ERROR `#[thread_local]` attribute cannot be used on closures
}

struct S {
    #[thread_local]
    //~^ ERROR `#[thread_local]` attribute cannot be used on struct fields
    a: String,
    b: String,
}

#[thread_local]
// Static. OK.
static B: u32 = 0;

extern "C" {
    #[thread_local]
    // Foreign static. OK.
    static C: u32;
}
