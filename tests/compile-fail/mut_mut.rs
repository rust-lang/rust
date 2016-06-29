#![feature(plugin)]
#![plugin(clippy)]

#![allow(unused, no_effect, unnecessary_operation)]
#![deny(mut_mut)]

//#![plugin(regex_macros)]
//extern crate regex;

fn fun(x : &mut &mut u32) -> bool { //~ERROR generally you want to avoid `&mut &mut
    **x > 0
}

fn less_fun(x : *mut *mut u32) {
  let y = x;
}

macro_rules! mut_ptr {
    ($p:expr) => { &mut $p }
    //~^ ERROR generally you want to avoid `&mut &mut
}

#[allow(unused_mut, unused_variables)]
fn main() {
    let mut x = &mut &mut 1u32; //~ERROR generally you want to avoid `&mut &mut
    {
        let mut y = &mut x; //~ERROR this expression mutably borrows a mutable reference
    }

    if fun(x) {
        let y : &mut &mut u32 = &mut &mut 2;
        //~^ ERROR generally you want to avoid `&mut &mut
        //~| ERROR generally you want to avoid `&mut &mut
        //~| ERROR generally you want to avoid `&mut &mut
        **y + **x;
    }

    if fun(x) {
        let y : &mut &mut &mut u32 = &mut &mut &mut 2;
        //~^ ERROR generally you want to avoid `&mut &mut
        //~| ERROR generally you want to avoid `&mut &mut
        //~| ERROR generally you want to avoid `&mut &mut
        //~| ERROR generally you want to avoid `&mut &mut
        //~| ERROR generally you want to avoid `&mut &mut
        //~| ERROR generally you want to avoid `&mut &mut
        ***y + **x;
    }

    let mut z = mut_ptr!(&mut 3u32);
    //~^ NOTE in this expansion of mut_ptr!
}

fn issue939() {
    let array = [5, 6, 7, 8, 9];
    let mut args = array.iter().skip(2);
    for &arg in &mut args {
        println!("{}", arg);
    }

    let args = &mut args;
    for arg in args {
        println!(":{}", arg);
    }
}
