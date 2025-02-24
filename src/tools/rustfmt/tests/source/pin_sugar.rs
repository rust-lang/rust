// See #130494

#![feature(pin_ergonomics)]
#![allow(incomplete_features)]

fn f(x: &pin  const i32) {}
fn g<'a>(x: &  'a pin const  i32) {}
fn h<'a>(x: &  'a pin  
mut i32) {}
fn i(x: &pin      mut  i32) {}

fn pinned_locals() {
    let    pin                 mut
x = 0_i32;
    let    pin    /* comment */             mut
x = 0_i32;
    let    pin                 mut /* comment */
x = 0_i32;
    let          pin
const 
y = 0_i32;
    let          pin
/* comment */ const 
y = 0_i32;
    let          pin
const /* comment */
y = 0_i32;
let (
    pin               
    const      x,
    pin           mut   y
) = (0_i32, 0_i32);
let (
    pin               /* comment */
    const      x,
    pin   /* comment */        mut   y
) = (0_i32, 0_i32);
let (
    pin               
    const  /* comment */    x,
    pin           mut  /* comment */ y
) = (0_i32, 0_i32);
}