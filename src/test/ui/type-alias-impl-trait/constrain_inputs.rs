// check-pass

#![feature(type_alias_impl_trait)]

mod foo {
    type Ty<'a> = impl Sized;
    fn defining(s: &str) -> Ty<'_> { s }
    fn execute(ty: Ty<'_>) -> &str { todo!() }
}

mod bar {
    type Ty<'a> = impl FnOnce() -> &'a str;
    fn defining(s: &str) -> Ty<'_> { move || s }
    fn execute(ty: Ty<'_>) -> &str { ty() }
}

fn main() {}
