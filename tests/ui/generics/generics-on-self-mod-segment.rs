struct Ty;

fn self_(_: self::<i32>::Ty) {}
//~^ ERROR type arguments are not allowed on module `generics_on_self_mod_segment`

fn crate_(_: crate::<i32>::Ty) {}
//~^ ERROR type arguments are not allowed on module `generics_on_self_mod_segment`

macro_rules! dollar_crate {
    () => {
        fn dollar_crate_(_: $crate::<i32>::Ty) {}
        //~^ ERROR type arguments are not allowed on module `generics_on_self_mod_segment`
    }
}

dollar_crate!();

fn main() {}
