// compile-pass
#![allow(unused)]

macro_rules! m {
    ($($e1:expr),*; $($e2:expr),*) => {
        $( let x = $e1 )*; //~ WARN expected `;`
        $( println!("{}", $e2) )*; //~ WARN expected `;`
    }
}


fn main() { m!(0, 0; 0, 0); }
