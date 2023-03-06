// revisions: local alias

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct Family<T>(T);

impl Family<()> {
    type Proj = ();
}

impl<T> Family<Result<T, ()>> {
    type Proj = Self;
}

#[cfg(alias)]
type Alias = Family<Option<()>>::Proj; //[alias]~ ERROR associated type `Proj` not found for `Family<Option<()>>`

fn main() {
    #[cfg(local)]
    let _: Family<std::path::PathBuf>::Proj = (); //[local]~ ERROR associated type `Proj` not found for `Family<PathBuf>`
}
