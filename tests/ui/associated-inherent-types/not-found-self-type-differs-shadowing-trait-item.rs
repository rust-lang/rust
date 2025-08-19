#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

// Check that it's okay to report “[inherent] associated type […] not found” for inherent associated
// type candidates that are not applicable (due to unsuitable Self type) even if there exists a
// “shadowed” associated type from a trait with the same name since its use would be ambiguous
// anyway if the IAT didn't exist.
// FIXME(inherent_associated_types): Figure out which error would be more helpful here.

//@ revisions: shadowed uncovered

struct S<T>(T);

trait Tr {
    type Pr;
}

impl<T> Tr for S<T> {
    type Pr = ();
}

#[cfg(shadowed)]
impl S<()> {
    type Pr = i32;
}

fn main() {
    let _: S::<bool>::Pr = ();
    //[shadowed]~^ ERROR associated type `Pr` not found
    //[uncovered]~^^ ERROR associated type `Pr` not found
}
