// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: ast mir
//[mir]compile-flags: -Z emit-end-regions -Z borrowck-mir

#![feature(advanced_slice_patterns)]

pub struct Foo {
  x: u32
}

pub struct Bar(u32);

pub enum Baz {
    X(u32)
}

union U {
    a: u8,
    b: u64,
}

impl Foo {
  fn x(&mut self) -> &mut u32 { &mut self.x }
}

impl Bar {
    fn x(&mut self) -> &mut u32 { &mut self.0 }
}

impl Baz {
    fn x(&mut self) -> &mut u32 {
        match *self {
            Baz::X(ref mut value) => value
        }
    }
}

static mut sfoo : Foo = Foo{x: 23 };
static mut sbar : Bar = Bar(23);
static mut stuple : (i32, i32) = (24, 25);
static mut senum : Baz = Baz::X(26);
static mut sunion : U = U { a: 0 };

fn main() {
    // Local and field from struct
    {
        let mut f = Foo { x: 22 };
        let _x = f.x();
        f.x; //[ast]~ ERROR cannot use `f.x` because it was mutably borrowed
             //[mir]~^ ERROR cannot use `f.x` because it was mutably borrowed (Ast)
             //[mir]~| ERROR cannot use `f.x` because it was mutably borrowed (Mir)
    }
    // Local and field from tuple-struct
    {
        let mut g = Bar(22);
        let _0 = g.x();
        g.0; //[ast]~ ERROR cannot use `g.0` because it was mutably borrowed
             //[mir]~^ ERROR cannot use `g.0` because it was mutably borrowed (Ast)
             //[mir]~| ERROR cannot use `g.0` because it was mutably borrowed (Mir)
    }
    // Local and field from tuple
    {
        let mut h = (22, 23);
        let _0 = &mut h.0;
        h.0; //[ast]~ ERROR cannot use `h.0` because it was mutably borrowed
             //[mir]~^ ERROR cannot use `h.0` because it was mutably borrowed (Ast)
             //[mir]~| ERROR cannot use `h.0` because it was mutably borrowed (Mir)
    }
    // Local and field from enum
    {
        let mut e = Baz::X(2);
        let _e0 = e.x();
        match e {
            Baz::X(value) => value
            //[ast]~^ ERROR cannot use `e.0` because it was mutably borrowed
            //[mir]~^^ ERROR cannot use `e.0` because it was mutably borrowed (Ast)
            //[mir]~| ERROR cannot use `e.0` because it was mutably borrowed (Mir)
        };
    }
    // Local and field from union
    unsafe {
        let mut u = U { b: 0 };
        let _ra = &mut u.a;
        u.a; //[ast]~ ERROR cannot use `u.a` because it was mutably borrowed
             //[mir]~^ ERROR cannot use `u.a` because it was mutably borrowed (Ast)
             //[mir]~| ERROR cannot use `u.a` because it was mutably borrowed (Mir)
    }
    // Static and field from struct
    unsafe {
        let _x = sfoo.x();
        sfoo.x; //[mir]~ ERROR cannot use `sfoo.x` because it was mutably borrowed (Mir)
    }
    // Static and field from tuple-struct
    unsafe {
        let _0 = sbar.x();
        sbar.0; //[mir]~ ERROR cannot use `sbar.0` because it was mutably borrowed (Mir)
    }
    // Static and field from tuple
    unsafe {
        let _0 = &mut stuple.0;
        stuple.0; //[mir]~ ERROR cannot use `stuple.0` because it was mutably borrowed (Mir)
    }
    // Static and field from enum
    unsafe {
        let _e0 = senum.x();
        match senum {
            Baz::X(value) => value
            //[mir]~^ ERROR cannot use `senum.0` because it was mutably borrowed (Mir)
        };
    }
    // Static and field from union
    unsafe {
        let _ra = &mut sunion.a;
        sunion.a; //[mir]~ ERROR cannot use `sunion.a` because it was mutably borrowed (Mir)
    }
    // Deref and field from struct
    {
        let mut f = Box::new(Foo { x: 22 });
        let _x = f.x();
        f.x; //[ast]~ ERROR cannot use `f.x` because it was mutably borrowed
             //[mir]~^ ERROR cannot use `f.x` because it was mutably borrowed (Ast)
             //[mir]~| ERROR cannot use `f.x` because it was mutably borrowed (Mir)
    }
    // Deref and field from tuple-struct
    {
        let mut g = Box::new(Bar(22));
        let _0 = g.x();
        g.0; //[ast]~ ERROR cannot use `g.0` because it was mutably borrowed
             //[mir]~^ ERROR cannot use `g.0` because it was mutably borrowed (Ast)
             //[mir]~| ERROR cannot use `g.0` because it was mutably borrowed (Mir)
    }
    // Deref and field from tuple
    {
        let mut h = Box::new((22, 23));
        let _0 = &mut h.0;
        h.0; //[ast]~ ERROR cannot use `h.0` because it was mutably borrowed
             //[mir]~^ ERROR cannot use `h.0` because it was mutably borrowed (Ast)
             //[mir]~| ERROR cannot use `h.0` because it was mutably borrowed (Mir)
    }
    // Deref and field from enum
    {
        let mut e = Box::new(Baz::X(3));
        let _e0 = e.x();
        match *e {
            Baz::X(value) => value
            //[ast]~^ ERROR cannot use `e.0` because it was mutably borrowed
            //[mir]~^^ ERROR cannot use `e.0` because it was mutably borrowed (Ast)
            //[mir]~| ERROR cannot use `e.0` because it was mutably borrowed (Mir)
        };
    }
    // Deref and field from union
    unsafe {
        let mut u = Box::new(U { b: 0 });
        let _ra = &mut u.a;
        u.a; //[ast]~ ERROR cannot use `u.a` because it was mutably borrowed
             //[mir]~^ ERROR cannot use `u.a` because it was mutably borrowed (Ast)
             //[mir]~| ERROR cannot use `u.a` because it was mutably borrowed (Mir)
    }
    // Constant index
    {
        let mut v = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let _v = &mut v;
        match v {
            &[x, _, .., _, _] => println!("{}", x),
                //[ast]~^ ERROR cannot use `v[..]` because it was mutably borrowed
                //[mir]~^^ ERROR cannot use `v[..]` because it was mutably borrowed (Ast)
                //[mir]~| ERROR cannot use `v[0]` because it was mutably borrowed (Mir)
                            _ => panic!("other case"),
        }
        match v {
            &[_, x, .., _, _] => println!("{}", x),
                //[ast]~^ ERROR cannot use `v[..]` because it was mutably borrowed
                //[mir]~^^ ERROR cannot use `v[..]` because it was mutably borrowed (Ast)
                //[mir]~| ERROR cannot use `v[1]` because it was mutably borrowed (Mir)
                            _ => panic!("other case"),
        }
        match v {
            &[_, _, .., x, _] => println!("{}", x),
                //[ast]~^ ERROR cannot use `v[..]` because it was mutably borrowed
                //[mir]~^^ ERROR cannot use `v[..]` because it was mutably borrowed (Ast)
                //[mir]~| ERROR cannot use `v[-2]` because it was mutably borrowed (Mir)
                            _ => panic!("other case"),
        }
        match v {
            &[_, _, .., _, x] => println!("{}", x),
                //[ast]~^ ERROR cannot use `v[..]` because it was mutably borrowed
                //[mir]~^^ ERROR cannot use `v[..]` because it was mutably borrowed (Ast)
                //[mir]~| ERROR cannot use `v[-1]` because it was mutably borrowed (Mir)
                            _ => panic!("other case"),
        }
    }
}
