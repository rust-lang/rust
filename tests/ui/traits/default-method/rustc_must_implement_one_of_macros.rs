#![feature(rustc_attrs)]

macro_rules! define_method {
    ($name:ident) => {
        fn $name() {}
    };
}

#[rustc_must_implement_one_of(a, b)]
trait TraitMacro {
    fn a() {}
    define_method!(b);
}

macro_rules! define_trait {
    ($trait_name:ident, $m1:ident, $m2:ident) => {
        #[rustc_must_implement_one_of($m1, $m2)]
        trait $trait_name {
            fn $m1() {}
            fn $m2() {}
        }
    };
}

define_trait!(TraitMacroWhole, foo, bar);

macro_rules! define_internal_b {
    () => {
        fn b() {}
    };
}

#[rustc_must_implement_one_of(a, b)]
//~^ ERROR function not found in this trait
trait TraitHygiene {
    fn a() {}
    define_internal_b!();
}

fn main() {}
