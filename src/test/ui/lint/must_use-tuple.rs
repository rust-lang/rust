#![deny(unused_must_use)]

fn foo() -> (Result<(), ()>, ()) {
    (Ok::<(), ()>(()), ())
}

fn main() {
    (Ok::<(), ()>(()),); //~ ERROR unused `std::result::Result`

    (Ok::<(), ()>(()), 0, Ok::<(), ()>(()), 5);
    //~^ ERROR unused `std::result::Result`
    //~^^ ERROR unused `std::result::Result`

    foo(); //~ ERROR unused `std::result::Result`

    ((Err::<(), ()>(()), ()), ()); //~ ERROR unused `std::result::Result`
}
