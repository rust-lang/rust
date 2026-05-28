// Test overloaded resolution of fn_traits.
//@ run-pass

#![feature(fn_traits)]
#![feature(unboxed_closures)]

#[derive(Debug, PartialEq, Eq)]
struct Ishmael;
#[derive(Debug, PartialEq, Eq)]
struct Maybe;
struct CallMe;

impl FnOnce<(Ishmael,)> for CallMe {
    type Output = Ishmael;
    extern "rust-call" fn call_once(self, _args: (Ishmael,)) -> Ishmael {
        println!("Split your lungs with blood and thunder!");
        Ishmael
    }
}

impl FnOnce<(Maybe,)> for CallMe {
    type Output = Maybe;
    extern "rust-call" fn call_once(self, _args: (Maybe,)) -> Maybe {
        println!("So we just met, and this is crazy");
        Maybe
    }
}

fn main() {
    assert_eq!(CallMe(Ishmael), Ishmael);
    assert_eq!(CallMe(Maybe), Maybe);
}
