#![feature(const_closures, const_trait_impl)]

trait Foo {
    fn foo(&self);
}

impl Foo for () {
    fn foo(&self) {}
}

fn main() {
    // #150052 deduplicate diagnostics for const trait supertraits
    // so we only get one error here
    (const || { (()).foo() })();
    //~^ ERROR: cannot use `const` closures outside of const contexts
}
