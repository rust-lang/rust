fn xyz() -> u8 { 42 }

const NUM: u8 = xyz();
//~^ ERROR calls in constants are limited to constant functions, tuple structs and tuple variants
//~| ERROR any use of this value will cause an error [const_err]

fn main() {
    match 1 {
        NUM => unimplemented!(),
        //~^ ERROR could not evaluate constant pattern
        _ => unimplemented!(),
    }
}
