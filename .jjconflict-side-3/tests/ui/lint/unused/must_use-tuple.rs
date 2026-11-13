#![deny(unused_must_use)]

fn foo() -> (Result<(), ()>, ()) {
    (Ok::<(), ()>(()), ())
}

fn main() {
    (Ok::<(), ()>(()),); //~ ERROR unused `Result`

    (Ok::<(), ()>(()), 0, Ok::<(), ()>(()), 5);
    //~^ ERROR unused `Result`
    //~^^ ERROR unused `Result`

    foo(); //~ ERROR unused `Result`

    ((Err::<(), ()>(()), ()), ()); //~ ERROR unused `Result`
}
