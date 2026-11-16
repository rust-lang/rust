// Typo of `CONST` to `CONS`. #149660

const CONST: usize = 0;

fn foo<const C: usize>() {}

fn main() {
    foo::<CONS>();
    //~^ ERROR cannot find type `CONS` in this scope
}
