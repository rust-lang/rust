// Test overloaded resolution of fn_traits.
// run-pass

#![feature(fn_traits)]
#![feature(unboxed_closures)]

struct Ishmael;
struct Maybe;
struct CallMe;

impl FnOnce<(Ishmael,)> for CallMe {
    type Output = ();
    extern "rust-call" fn call_once(self, _args: (Ishmael,)) -> () {
        println!("Split your lungs with blood and thunder!");
    }
}

impl FnOnce<(Maybe,)> for CallMe {
    type Output = ();
    extern "rust-call" fn call_once(self, _args: (Maybe,)) -> () {
        println!("So we just met, and this is crazy");
    }
}

fn main() {
    CallMe(Ishmael);
    CallMe(Maybe);
}
