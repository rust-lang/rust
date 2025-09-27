//@ edition: 2024

macro_rules! foo {
    () => {
        #[cfg($crate)] //~ ERROR `cfg` predicate key must be an identifier
        #[cfg_attr($crate, path = "foo")] //~ ERROR `cfg` predicate key must be an identifier
        mod _x {}

        cfg!($crate); //~ ERROR `cfg` predicate key must be an identifier
    };
}

#[cfg(crate)] //~ ERROR `cfg` predicate key must be an identifier
#[cfg(super)] //~ ERROR `cfg` predicate key must be an identifier
#[cfg(self)] //~ ERROR `cfg` predicate key must be an identifier
#[cfg(Self)] //~ ERROR `cfg` predicate key must be an identifier
#[cfg_attr(crate, path = "foo")] //~ ERROR `cfg` predicate key must be an identifier
#[cfg_attr(super, path = "foo")] //~ ERROR `cfg` predicate key must be an identifier
#[cfg_attr(self, path = "foo")] //~ ERROR `cfg` predicate key must be an identifier
#[cfg_attr(Self, path = "foo")] //~ ERROR `cfg` predicate key must be an identifier
mod _path_kw {}

#[cfg(struct)] //~ ERROR expected identifier, found keyword
//~^ WARNING unexpected `cfg` condition name
mod _non_path_kw1 {}
#[cfg(enum)] //~ ERROR expected identifier, found keyword
//~^ WARNING unexpected `cfg` condition name
mod _non_path_kw2 {}
#[cfg(async)] //~ ERROR expected identifier, found keyword
//~^ WARNING unexpected `cfg` condition name
mod _non_path_kw3 {}
#[cfg(impl)] //~ ERROR expected identifier, found keyword
//~^ WARNING unexpected `cfg` condition name
mod _non_path_kw4 {}
#[cfg(trait)] //~ ERROR expected identifier, found keyword
//~^ WARNING unexpected `cfg` condition name
mod _non_path_kw5 {}
#[cfg_attr(struct, path = "foo")] //~ ERROR expected identifier, found keyword
//~^ WARNING unexpected `cfg` condition name
mod _non_path_kw6 {}
#[cfg_attr(enum, path = "foo")] //~ ERROR expected identifier, found keyword
//~^ WARNING unexpected `cfg` condition name
mod _non_path_kw7 {}
#[cfg_attr(async, path = "foo")] //~ ERROR expected identifier, found keyword
//~^ WARNING unexpected `cfg` condition name
mod _non_path_kw8 {}
#[cfg_attr(impl, path = "foo")] //~ ERROR expected identifier, found keyword
//~^ WARNING unexpected `cfg` condition name
mod _non_path_kw9 {}
#[cfg_attr(trait, path = "foo")] //~ ERROR expected identifier, found keyword
//~^ WARNING unexpected `cfg` condition name
mod _non_path_kw10 {}

fn main() {
    foo!();

    cfg!(crate); //~ ERROR `cfg` predicate key must be an identifier
    cfg!(super); //~ ERROR `cfg` predicate key must be an identifier
    cfg!(self); //~ ERROR `cfg` predicate key must be an identifier
    cfg!(Self); //~ ERROR `cfg` predicate key must be an identifier

    cfg!(struct); //~ ERROR expected identifier, found keyword
    //~^ WARNING unexpected `cfg` condition name
    cfg!(enum); //~ ERROR expected identifier, found keyword
    //~^ WARNING unexpected `cfg` condition name
    cfg!(async); //~ ERROR expected identifier, found keyword
    //~^ WARNING unexpected `cfg` condition name
    cfg!(impl); //~ ERROR expected identifier, found keyword
    //~^ WARNING unexpected `cfg` condition name
    cfg!(trait); //~ ERROR expected identifier, found keyword
    //~^ WARNING unexpected `cfg` condition name
}
