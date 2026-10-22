//@ check-pass

#![feature(generic_const_exprs)]
#![allow(incomplete_features, dead_code)]

trait Config {
    const N: usize = 4;
}

struct Concrete;
impl Config for Concrete {}

struct Registry<const N: usize>;

impl<const N: usize> Default for Registry<N> {
    fn default() -> Self {
        Self
    }
}

struct ConfiguredRegistry<C: Config>(std::marker::PhantomData<C>);

impl<C: Config> ConfiguredRegistry<C>
where
    [(); C::N]:,
{
    fn new(_: Registry<{ C::N }>) -> Self {
        Self(std::marker::PhantomData)
    }
}

fn reproduce() -> ConfiguredRegistry<Concrete> {
    let registry = Registry::default();
    ConfiguredRegistry::new(registry)
}

fn main() {}
