// compile-flags: -Zforce-overflow-checks=on

// these errors are not actually "const_err", they occur in codegen/consts
// and are unconditional warnings that can't be denied or allowed

#![allow(exceeding_bitshifts)]
#![allow(const_err)]

fn black_box<T>(_: T) {
    unimplemented!()
}

// Make sure that the two uses get two errors.
const FOO: u8 = [5u8][1];
//~^ ERROR constant evaluation error
//~| index out of bounds: the len is 1 but the index is 1

fn main() {
    black_box((FOO, FOO));
    //~^ ERROR referenced constant has errors
    //~| ERROR could not evaluate constant
}
