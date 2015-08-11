#![feature(plugin)]
#![plugin(clippy)]

//#![plugin(regex_macros)]
//extern crate regex;

#[deny(mut_mut)]
fn fun(x : &mut &mut u32) -> bool { //~ERROR
    **x > 0
}

macro_rules! mut_ptr {
    ($p:expr) => { &mut $p }
}

#[deny(mut_mut)]
#[allow(unused_mut, unused_variables)]
fn main() {
    let mut x = &mut &mut 1u32; //~ERROR
    {
        let mut y = &mut x; //~ERROR
    }

    if fun(x) {
        let y : &mut &mut &mut u32 = &mut &mut &mut 2;
                 //~^ ERROR
                      //~^^ ERROR
                                      //~^^^ ERROR
                                           //~^^^^ ERROR
        ***y + **x;
    }

    let mut z = mut_ptr!(&mut 3u32); //~ERROR
}
