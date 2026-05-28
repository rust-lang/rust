//@ revisions: real pre_expansion
//@[pre_expansion] check-pass

#[cfg(real)]
fn hello<'a>() -> impl Sized + use<'a> + use<'a> {}
//[real]~^ ERROR duplicate `use<...>` precise capturing syntax

fn main() {}
