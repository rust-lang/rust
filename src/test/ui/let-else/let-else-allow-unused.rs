// check-pass
// issue #89807

#![feature(let_else)]

#[deny(unused_variables)]

fn main() {
    let value = Some(String::new());
    #[allow(unused)]
    let banana = 1;
    #[allow(unused)]
    let Some(chaenomeles) = value else { return }; // OK
}
