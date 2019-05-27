#![deny(unused_must_use)]

fn foo() -> Result<(), ()> {
    Ok::<(), ()>(())
}

fn main() {
    (Ok::<(), ()>(()),); //~ ERROR unused `std::result::Result` that must be used

    (Ok::<(), ()>(()), 0, Ok::<(), ()>(()), 5);
    //~^ ERROR unused `std::result::Result` that must be used
    //~^^ ERROR unused `std::result::Result` that must be used

    foo(); //~ ERROR unused `std::result::Result` that must be used
}
