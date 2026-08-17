//! We want to reserve rights to be able to optimize statics declared as immutable,
//! so we defensively disallow immutable statics pointing to mutable allocations.

#[export_name = "S"]
static mut BACKING_S: i32 = 42;

fn main() {
    extern "C" {
        static S: i32;
    }
    let _val = &raw const S;
    //~^ ERROR: is declared as an immutable `static`, but the backing static is mutable
}
