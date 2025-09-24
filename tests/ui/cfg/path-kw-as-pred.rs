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
mod _y {}

fn main() {
    foo!();

    cfg!(crate); //~ ERROR `cfg` predicate key must be an identifier
    cfg!(super); //~ ERROR `cfg` predicate key must be an identifier
    cfg!(self); //~ ERROR `cfg` predicate key must be an identifier
    cfg!(Self); //~ ERROR `cfg` predicate key must be an identifier
}
