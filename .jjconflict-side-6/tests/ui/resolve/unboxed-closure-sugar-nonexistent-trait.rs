fn f<F:Nonexist(isize) -> isize>(x: F) {}
//~^ ERROR cannot find trait `Nonexist`

type Typedef = isize;

fn g<F:Typedef(isize) -> isize>(x: F) {}
//~^ ERROR expected trait, found type alias `Typedef`

fn main() {}
