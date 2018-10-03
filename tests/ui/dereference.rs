#![feature(tool_lints)]

use std::ops::{Deref, DerefMut};

#[allow(clippy::many_single_char_names, clippy::clone_double_ref)]
#[allow(unused_variables)]
#[warn(clippy::explicit_deref_method)]
fn main() {
    let a: &mut String = &mut String::from("foo");

    // these should require linting
    {
        let b: &str = a.deref();
    }

    {
        let b: &mut str = a.deref_mut();
    }

    {
        let b: String = a.deref().clone();
    }
    
    {
        let b: usize = a.deref_mut().len();
    }
    
    {
        let b: &usize = &a.deref().len();
    }

    {
        // only first deref should get linted here
        let b: &str = a.deref().deref();
    }

    {
        // both derefs should get linted here
        let b: String = format!("{}, {}", a.deref(), a.deref());
    }

    // these should not require linting
    {
        let b: &str = &*a;
    }

    {
        let b: &mut str = &mut *a;
    }

    {
        macro_rules! expr_deref { ($body:expr) => { $body.deref() } }
        let b: &str = expr_deref!(a);
    }
}
