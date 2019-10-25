fn main() {
    let box x = Box::new('c'); //~ ERROR box pattern syntax is experimental
    println!("x: {}", x);
}

macro_rules! accept_pat { ($p:pat) => {} }
accept_pat!(box 0); //~ ERROR box pattern syntax is experimental
