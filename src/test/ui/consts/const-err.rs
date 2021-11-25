// build-fail
// compile-flags: -C overflow-checks=on

#![allow(arithmetic_overflow)]
#![warn(const_err)]

fn black_box<T>(_: T) {
    unimplemented!()
}

const FOO: u8 = [5u8][1];
//~^ WARN any use of this value will cause an error
//~| WARN this was previously accepted by the compiler but is being phased out

fn main() {
    black_box((FOO, FOO));
    //~^ ERROR erroneous constant used
    //~| ERROR erroneous constant
}
