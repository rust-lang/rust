#![allow(unused_variables, clippy::many_single_char_names, clippy::clone_double_ref)]
#![warn(clippy::explicit_deref_method)]

use std::ops::{Deref, DerefMut};

fn concat(deref_str: &str) -> String {
    format!("{}bar", deref_str)
}

fn just_return(deref_str: &str) -> &str {
    deref_str
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

    // following should not require linting

    let b = just_return(a).deref();

    let b: String = concat(just_return(a).deref());

    let b: String = a.deref().clone();

    let b: usize = a.deref_mut().len();

    let b: &usize = &a.deref().len();

    let b: &str = a.deref().deref();

    let b: &str = &*a;

    let b: &mut str = &mut *a;

    macro_rules! expr_deref {
        ($body:expr) => {
            $body.deref()
        };
    }
    let b: &str = expr_deref!(a);

    let opt_a = Some(a);
    let b = opt_a.unwrap().deref();
}
