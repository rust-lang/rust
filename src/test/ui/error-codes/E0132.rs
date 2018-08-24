#![feature(start)]

#[start]
fn f< T >() {} //~ ERROR E0132

fn main() {
}
