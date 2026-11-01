//@ run-rustfix
//@ check-pass
#![deny(explicit_outlives_requirements)]

pub trait TypeCx {
    type Ty;
}

pub struct Pat<Cx: TypeCx> {
    pub ty: Cx::Ty,
}

// Simple recursive case: no warning
pub struct MyTypeContextSimpleRecursive<'thir, 'tcx: 'thir> {
    pub pat: Pat<MyTypeContextSimpleRecursive<'thir, 'tcx>>,
}
impl<'thir, 'tcx: 'thir> TypeCx for MyTypeContextSimpleRecursive<'thir, 'tcx> {
    type Ty = ();
}

// Non-recursive case: we want a warning
pub struct MyTypeContextNotRecursive<'thir, 'tcx: 'thir> {
    pub tcx: &'tcx (),
    pub thir: &'thir (),
}
impl<'thir, 'tcx: 'thir> TypeCx for MyTypeContextNotRecursive<'thir, 'tcx> {
    type Ty = ();
}


// Mixed-recursive case: we want a warning
pub struct MyTypeContextMixedRecursive<'thir, 'tcx: 'thir> {
    pub pat: Pat<MyTypeContextMixedRecursive<'thir, 'tcx>>,
    pub tcx: &'tcx (),
    pub thir: &'thir (),
}
impl<'thir, 'tcx: 'thir> TypeCx for MyTypeContextMixedRecursive<'thir, 'tcx> {
    type Ty = ();
}

fn main() {}
