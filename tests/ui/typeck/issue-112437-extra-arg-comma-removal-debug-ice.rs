#![feature(never_type)]

fn f(a: !) {}

fn main() {
    f(panic!(), 1);
    //~^ ERROR this function takes 1 argument but 2 arguments were supplied
}
