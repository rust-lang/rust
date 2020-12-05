fn xyz() -> u8 { 42 }

const NUM: u8 = xyz();
//~^ ERROR calls in constants are limited to constant functions, tuple structs and tuple variants

fn main() {
    match 1 {
        NUM => unimplemented!(),
        //~^ ERROR could not evaluate constant pattern
        //~| ERROR could not evaluate constant pattern
        _ => unimplemented!(),
    }
}
