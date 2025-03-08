#![feature(lazy_type_alias)]
//~^ WARNING: the feature `lazy_type_alias` is incomplete and may not be safe to use and/or cause compiler crashes [incomplete_features]

trait TraitA {}
trait TraitB {}

fn this(_x: &()) {}

fn needs_copy<T>() {
    type F<T>
    where
        //~^ ERROR: where clauses are not allowed before the type for type aliases
        #[cfg(a)]
        //~^ ERROR: attributes in `where` clause are unstable
        //~| WARNING: unexpected `cfg` condition name: `a`
        T: TraitA,
        #[cfg(b)]
        //~^ ERROR: attributes in `where` clause are unstable
        //~| WARNING: unexpected `cfg` condition name: `b`
        T: TraitB,
        ():,
        ():,
    = T;
    let x: () = ();
    this(&x);
}

fn main() {
    needs_copy::<()>();
}
