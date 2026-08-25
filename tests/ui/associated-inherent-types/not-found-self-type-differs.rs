#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct Family<T>(T);

impl Family<()> {
    type Proj = ();
}

impl<T> Family<Result<T, ()>> {
    type Proj = Self;
}

fn main() {
    let _: Family<Option<()>>::Proj; //~ ERROR associated type `Proj` not found for `Family<Option<()>>`
    let _: Family<std::path::PathBuf>::Proj = (); //~ ERROR associated type `Proj` not found for `Family<PathBuf>`
}
