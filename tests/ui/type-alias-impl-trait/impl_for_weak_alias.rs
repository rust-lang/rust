#![feature(type_alias_impl_trait)]
#![feature(rustc_attrs)]

type Alias = (impl Sized, u8);

#[rustc_auto_trait]
trait Trait {}
impl Trait for Alias {}
//~^ ERROR traits with a default impl, like `Trait`, cannot be implemented for type alias `Alias`

fn _def() -> Alias {
    (42, 42)
}

fn main() {}
