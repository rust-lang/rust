// issue #89807

#![feature(let_else)]

#[deny(unused_variables)]

fn main() {
    let value = Some(5);
    #[allow(unused)]
    let banana = 1;
    #[allow(unused)]
    let Some(chaenomeles) = value else { return }; // OK
    let Some(unused) = value else { return };
    //~^ ERROR unused variable: `unused`
}
