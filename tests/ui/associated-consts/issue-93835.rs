#![feature(type_ascription)]

fn e() {
    type_ascribe!(p, a<p:p<e=6>>);
    //~^ ERROR cannot find type `a`
    //~| ERROR cannot find value
    //~| ERROR associated const equality
    //~| ERROR cannot find trait `p`
}

fn main() {}
