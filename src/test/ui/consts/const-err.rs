// compile-flags: -Zforce-overflow-checks=on

#![allow(exceeding_bitshifts)]
#![warn(const_err)]

fn black_box<T>(_: T) {
    unimplemented!()
}

const FOO: u8 = [5u8][1];
//~^ WARN any use of this value will cause an error

fn main() {
    black_box((FOO, FOO));
    //~^ ERROR erroneous constant used
    //~| ERROR erroneous constant
}
