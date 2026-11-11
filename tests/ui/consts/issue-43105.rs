fn xyz() -> u8 { 42 }

const NUM: u8 = xyz();
//~^ ERROR cannot call non-const function

fn main() {
    match 1 {
        NUM => unimplemented!(), // ok, the `const` already emitted an error
        _ => unimplemented!(),
    }
}
