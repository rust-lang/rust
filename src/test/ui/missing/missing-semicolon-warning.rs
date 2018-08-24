#![feature(rustc_attrs)]
#![allow(unused)]

macro_rules! m {
    ($($e1:expr),*; $($e2:expr),*) => {
        $( let x = $e1 )*; //~ WARN expected `;`
        $( println!("{}", $e2) )*; //~ WARN expected `;`
    }
}

#[rustc_error]
fn main() { m!(0, 0; 0, 0); } //~ ERROR compilation successful
