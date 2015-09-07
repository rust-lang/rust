#![feature(plugin)]
#![plugin(clippy)]

#![allow(unused)]

//#![plugin(regex_macros)]
//extern crate regex;

#[deny(mut_mut)]
fn fun(x : &mut &mut u32) -> bool { //~ERROR generally you want to avoid `&mut &mut
    **x > 0
}

#[deny(mut_mut)]
fn less_fun(x : *mut *mut u32) {
  let y = x;
}

macro_rules! mut_ptr {
    ($p:expr) => { &mut $p }
}

#[deny(mut_mut)]
#[allow(unused_mut, unused_variables)]
fn main() {
    let mut x = &mut &mut 1u32; //~ERROR generally you want to avoid `&mut &mut
    {
        let mut y = &mut x; //~ERROR this expression mutably borrows a mutable reference
    }

    if fun(x) {
        let y : &mut &mut &mut u32 = &mut &mut &mut 2;
                 //~^ ERROR generally you want to avoid `&mut &mut
                      //~^^ ERROR generally you want to avoid `&mut &mut
                                      //~^^^ ERROR generally you want to avoid `&mut &mut
                                           //~^^^^ ERROR generally you want to avoid `&mut &mut
        ***y + **x;
    }

    let mut z = mut_ptr!(&mut 3u32); //~ERROR generally you want to avoid `&mut &mut
}
