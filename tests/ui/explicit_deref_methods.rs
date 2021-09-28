// run-rustfix

#![allow(unused_variables, clippy::clone_double_ref)]
#![warn(clippy::explicit_deref_methods)]

use std::ops::{Deref, DerefMut};

fn concat(deref_str: &str) -> String {
    format!("{}bar", deref_str)
}

fn just_return(deref_str: &str) -> &str {
    deref_str
}

struct CustomVec(Vec<u8>);
impl Deref for CustomVec {
    type Target = Vec<u8>;

    fn deref(&self) -> &Vec<u8> {
        &self.0
    }
}

fn main() {
    let a: &mut String = &mut String::from("foo");

    // these should require linting

    let b: &str = a.deref();

    let b: &mut str = a.deref_mut();

    // both derefs should get linted here
    let b: String = format!("{}, {}", a.deref(), a.deref());

    println!("{}", a.deref());

    #[allow(clippy::match_single_binding)]
    match a.deref() {
        _ => (),
    }

    let b: String = concat(a.deref());

    let b = just_return(a).deref();

    let b: String = concat(just_return(a).deref());

    let b: &str = a.deref().deref();

    let opt_a = Some(a.clone());
    let b = opt_a.unwrap().deref();

    // following should not require linting

    let cv = CustomVec(vec![0, 42]);
    let c = cv.deref()[0];

    let b: &str = &*a.deref();

    let b: String = a.deref().clone();

    let b: usize = a.deref_mut().len();

    let b: &usize = &a.deref().len();

    let b: &str = &*a;

    let b: &mut str = &mut *a;

    macro_rules! expr_deref {
        ($body:expr) => {
            $body.deref()
        };
    }
    let b: &str = expr_deref!(a);

    let b: &str = expr_deref!(a.deref());

    // The struct does not implement Deref trait
    #[derive(Copy, Clone)]
    struct NoLint(u32);
    impl NoLint {
        pub fn deref(self) -> u32 {
            self.0
        }
        pub fn deref_mut(self) -> u32 {
            self.0
        }
    }
    let no_lint = NoLint(42);
    let b = no_lint.deref();
    let b = no_lint.deref_mut();
}
