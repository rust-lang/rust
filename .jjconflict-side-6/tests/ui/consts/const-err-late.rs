//@ build-fail
//@ compile-flags: -C overflow-checks=on
//@ dont-require-annotations: NOTE

#![allow(arithmetic_overflow, unconditional_panic)]

fn black_box<T>(_: T) {
    unimplemented!()
}

struct S<T>(T);

impl<T> S<T> {
    const FOO: u8 = [5u8][1];
    //~^ ERROR index out of bounds: the length is 1 but the index is 1
    //~| ERROR index out of bounds: the length is 1 but the index is 1
}

fn main() {
    black_box((S::<i32>::FOO, S::<u32>::FOO)); //~ NOTE constant
}
