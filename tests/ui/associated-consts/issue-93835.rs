#![feature(type_ascription)]

fn e() {
    type_ascribe!(p, a<p:p<e=6>>);
    //~^ ERROR cannot find type `a` in this scope
    //~| ERROR cannot find value
    //~| ERROR associated const equality
    //~| ERROR cannot find trait `p` in this scope
    //~| ERROR associated const equality
}

fn main() {}
