//! We want to reserve rights to be able to inject implicit writes to mutable declared statics,
//! so we defensively disallow mutable statics pointing to immutable allocations.

#[export_name = "S"]
static IMMUT_S: i32 = 42;

fn main() {
    extern "C" {
        static mut S: i32;
    }
    let _val = &raw const S;
    //~^ ERROR: is declared as an mutable `static`, but the backing static is immutable
}
