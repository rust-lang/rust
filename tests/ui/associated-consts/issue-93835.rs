#![feature(type_ascription)]

fn e() {
    type_ascribe!(p, a<p:p<e=6>>);
    //~^ ERROR cannot find type `a` in this scope
    //~| ERROR path separator must be a double colon
    //~| ERROR cannot find value
    //~| ERROR associated const equality
    //~| ERROR associated const equality
    //~| ERROR failed to resolve: use of unresolved module or unlinked crate `p`
}

fn main() {}
