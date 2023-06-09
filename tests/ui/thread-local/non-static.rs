// Check that #[thread_local] attribute is rejected on non-static items.
#![feature(thread_local)]

#[thread_local]
//~^ ERROR attribute should be applied to a static
const A: u32 = 0;

#[thread_local]
//~^ ERROR attribute should be applied to a static
fn main() {
    #[thread_local] || {};
    //~^ ERROR attribute should be applied to a static
}

struct S {
    #[thread_local]
    //~^ ERROR attribute should be applied to a static
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
