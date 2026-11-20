#![feature(generic_const_exprs)]

trait B {
    type U<T>;
}

fn f<T: B<U<1i32> = ()>>() {}
//~^ ERROR constant provided when a type was expected

fn main() {}
