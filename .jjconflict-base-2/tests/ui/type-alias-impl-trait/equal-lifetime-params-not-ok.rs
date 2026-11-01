// issue: #112841

#![feature(type_alias_impl_trait)]

trait Trait<'a, 'b> {}
impl<T> Trait<'_, '_> for T {}

mod mod1 {
    type Opaque<'a, 'b> = impl super::Trait<'a, 'b>;
    #[define_opaque(Opaque)]
    fn test<'a>() -> Opaque<'a, 'a> {}
    //~^ ERROR non-defining opaque type use in defining scope
}

mod mod2 {
    type Opaque<'a, 'b> = impl super::Trait<'a, 'b>;
    #[define_opaque(Opaque)]
    fn test<'a: 'b, 'b: 'a>() -> Opaque<'a, 'b> {}
    //~^ ERROR non-defining opaque type use in defining scope
}

mod mod3 {
    type Opaque<'a, 'b> = impl super::Trait<'a, 'b>;
    #[define_opaque(Opaque)]
    fn test<'a: 'b, 'b: 'a>(a: &'a str) -> Opaque<'a, 'b> { a }
    //~^ ERROR non-defining opaque type use in defining scope
}

// This is similar to the previous cases in that 'a is equal to 'static,
// which is some sense an implicit parameter to `Opaque`.
// For example, given a defining use `Opaque<'a> := &'a ()`,
// it is ambiguous whether `Opaque<'a> := &'a ()` or `Opaque<'a> := &'static ()`
mod mod4 {
    type Opaque<'a> = impl super::Trait<'a, 'a>;
    #[define_opaque(Opaque)]
    fn test<'a: 'static>() -> Opaque<'a> {}
    //~^ ERROR expected generic lifetime parameter, found `'static`
}

fn main() {}
