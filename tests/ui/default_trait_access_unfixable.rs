//@no-rustfix: the trimmed replacement path may not be in scope
#![warn(clippy::default_trait_access)]

fn main() {
    let _: std::time::Duration = Default::default();
    //~^ default_trait_access
}
