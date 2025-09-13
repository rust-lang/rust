// Traits where a method is c-variadic are not dyn compatible.
//
// Creating a function pointer from a method on an `&dyn T` value creates a ReifyShim.
// This shim cannot reliably forward C-variadic arguments. Thus the trait as a whole
// is dyn-incompatible to prevent invalid shims from being created.
#![feature(c_variadic)]

#[repr(transparent)]
struct Struct(u64);

trait Trait {
    fn get(&self) -> u64;

    unsafe extern "C" fn dyn_method_ref(&self, mut ap: ...) -> u64 {
        self.get() + unsafe { ap.arg::<u64>() }
    }
}

impl Trait for Struct {
    fn get(&self) -> u64 {
        self.0
    }
}

fn main() {
    unsafe {
        let dyn_object: &dyn Trait = &Struct(64);
        //~^ ERROR the trait `Trait` is not dyn compatible
        assert_eq!(dyn_object.dyn_method_ref(100), 164);
        assert_eq!(
            (Trait::dyn_method_ref as unsafe extern "C" fn(_, ...) -> u64)(dyn_object, 100),
            164
        );
    }
}
