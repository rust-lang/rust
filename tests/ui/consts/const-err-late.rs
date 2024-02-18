//@ build-fail
//@ compile-flags: -C overflow-checks=on

#![allow(arithmetic_overflow, unconditional_panic)]

fn black_box<T>(_: T) {
    unimplemented!()
}

struct S<T>(T);

impl<T> S<T> {
    const FOO: u8 = [5u8][1];
    //~^ ERROR evaluation of `S::<i32>::FOO` failed
    //~| ERROR evaluation of `S::<u32>::FOO` failed
}

fn main() {
    black_box((S::<i32>::FOO, S::<u32>::FOO)); //~ constant
}
