fn xyz() -> u8 { 42 }

const NUM: u8 = xyz();
//~^ ERROR cannot call non-const fn

fn main() {
    match 1 {
        NUM => unimplemented!(),
        //~^ ERROR could not evaluate constant pattern
        //~| ERROR could not evaluate constant pattern
        _ => unimplemented!(),
    }
}
