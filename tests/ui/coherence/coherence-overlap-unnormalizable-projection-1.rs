// Regression test for soundness issue #114061:
// "Coherence incorrectly considers `unnormalizable_projection: Trait` to not hold even if it could"
#![crate_type = "lib"]

pub trait WhereBound {}
impl WhereBound for () {}

pub trait WithAssoc<'a> {
    type Assoc;
}

// These two impls of `Trait` overlap:

pub trait Trait {}
impl<T> Trait for T
where
    T: 'static,
    for<'a> T: WithAssoc<'a>,
    for<'a> Box<<T as WithAssoc<'a>>::Assoc>: WhereBound,
{
}

impl<T> Trait for Box<T> {} //~ ERROR conflicting implementations of trait `Trait` for type `Box<_>`

// A downstream crate could write:
//
//
//     use upstream::*;
//
//     struct Local;
//     impl WithAssoc<'_> for Box<Local> {
//         type Assoc = Local;
//     }
//
//     impl WhereBound for Box<Local> {}
//
//     fn impls_trait<T: Trait>() {}
//
//     fn main() {
//         impls_trait::<Box<Local>>();
//     }
