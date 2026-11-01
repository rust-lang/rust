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

fn main() {
    // Local and field from struct
    {
        let mut f = Foo { x: 22 };
        let x = f.x();
        f.x; //~ ERROR cannot use `f.x` because it was mutably borrowed
        drop(x);
    }
    // Local and field from tuple-struct
    {
        let mut g = Bar(22);
        let x = g.x();
        g.0; //~ ERROR cannot use `g.0` because it was mutably borrowed
        drop(x);
    }
    // Local and field from tuple
    {
        let mut h = (22, 23);
        let x = &mut h.0;
        h.0; //~ ERROR cannot use `h.0` because it was mutably borrowed
        drop(x);
    }
    // Local and field from enum
    {
        let mut e = Baz::X(2);
        let x = e.x();
        match e {
            Baz::X(value) => value //~ ERROR cannot use `e.0` because it was mutably borrowed
        };
        drop(x);
    }
    // Local and field from union
    unsafe {
        let mut u = U { b: 0 };
        let x = &mut u.a;
        u.a; //~ ERROR cannot use `u.a` because it was mutably borrowed
        drop(x);
    }
    // Deref and field from struct
    {
        let mut f = Box::new(Foo { x: 22 });
        let x = f.x();
        f.x; //~ ERROR cannot use `f.x` because it was mutably borrowed
        drop(x);
    }
    // Deref and field from tuple-struct
    {
        let mut g = Box::new(Bar(22));
        let x = g.x();
        g.0; //~ ERROR cannot use `g.0` because it was mutably borrowed
        drop(x);
    }
    // Deref and field from tuple
    {
        let mut h = Box::new((22, 23));
        let x = &mut h.0;
        h.0; //~ ERROR cannot use `h.0` because it was mutably borrowed
        drop(x);
    }
    // Deref and field from enum
    {
        let mut e = Box::new(Baz::X(3));
        let x = e.x();
        match *e {
            Baz::X(value) => value
            //~^ ERROR cannot use `e.0` because it was mutably borrowed
        };
        drop(x);
    }
    // Deref and field from union
    unsafe {
        let mut u = Box::new(U { b: 0 });
        let x = &mut u.a;
        u.a; //~ ERROR cannot use `u.a` because it was mutably borrowed
        drop(x);
    }
    // Constant index
    {
        let mut v = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let x = &mut v;
        match v {
            &[x, _, .., _, _] => println!("{}", x),
                //~^ ERROR cannot use `v[..]` because it was mutably borrowed
                            _ => panic!("other case"),
        }
        match v {
            &[_, x, .., _, _] => println!("{}", x),
                //~^ ERROR cannot use `v[..]` because it was mutably borrowed
                            _ => panic!("other case"),
        }
        match v {
            &[_, _, .., x, _] => println!("{}", x),
                //~^ ERROR cannot use `v[..]` because it was mutably borrowed
                            _ => panic!("other case"),
        }
        match v {
            &[_, _, .., _, x] => println!("{}", x),
                //~^ ERROR cannot use `v[..]` because it was mutably borrowed
                            _ => panic!("other case"),
        }
        drop(x);
    }
    // Subslices
    {
        let mut v = &[1, 2, 3, 4, 5];
        let x = &mut v;
        match v {
            &[x @ ..] => println!("{:?}", x),
                //~^ ERROR cannot use `v[..]` because it was mutably borrowed
            _ => panic!("other case"),
        }
        match v {
            &[_, x @ ..] => println!("{:?}", x),
                //~^ ERROR cannot use `v[..]` because it was mutably borrowed
            _ => panic!("other case"),
        }
        match v {
            &[x @ .., _] => println!("{:?}", x),
                //~^ ERROR cannot use `v[..]` because it was mutably borrowed
            _ => panic!("other case"),
        }
        match v {
            &[_, x @ .., _] => println!("{:?}", x),
                //~^ ERROR cannot use `v[..]` because it was mutably borrowed
            _ => panic!("other case"),
        }
        drop(x);
    }
    // Downcasted field
    {
        enum E<X> { A(X), B { x: X } }

        let mut e = E::A(3);
        let x = &mut e;
        match e {
            //~^ ERROR cannot use `e` because it was mutably borrowed
            E::A(ref ax) =>
                //~^ ERROR cannot borrow `e.0` as immutable because it is also borrowed as mutable
                println!("e.ax: {:?}", ax),
            E::B { x: ref bx } =>
                //~^ ERROR cannot borrow `e.x` as immutable because it is also borrowed as mutable
                println!("e.bx: {:?}", bx),
        }
        drop(x);
    }
    // Field in field
    {
        struct F { x: u32, y: u32 };
        struct S { x: F, y: (u32, u32), };
        let mut s = S { x: F { x: 1, y: 2}, y: (999, 998) };
        let x = &mut s;
        match s {
            S  { y: (ref y0, _), .. } =>
                //~^ ERROR cannot borrow `s.y.0` as immutable because it is also borrowed as mutable
                println!("y0: {:?}", y0),
            _ => panic!("other case"),
        }
        match s {
            S  { x: F { y: ref x0, .. }, .. } =>
                //~^ ERROR cannot borrow `s.x.y` as immutable because it is also borrowed as mutable
                println!("x0: {:?}", x0),
            _ => panic!("other case"),
        }
        drop(x);
    }
    // Field of ref
    {
        struct Block<'a> {
            current: &'a u8,
            unrelated: &'a u8,
        };

        fn bump<'a>(mut block: &mut Block<'a>) {
            let x = &mut block;
            let p: &'a u8 = &*block.current;
            //~^ ERROR cannot borrow `*block.current` as immutable because it is also borrowed as mutable
            // See issue rust#38899
            drop(x);
        }
    }
    // Field of ptr
    {
        struct Block2 {
            current: *const u8,
            unrelated: *const u8,
        }

        unsafe fn bump2(mut block: *mut Block2) {
            let x = &mut block;
            let p : *const u8 = &*(*block).current;
            //~^ ERROR cannot borrow `*block.current` as immutable because it is also borrowed as mutable
            // See issue rust#38899
            drop(x);
        }
    }
    // Field of index
    {
        struct F {x: u32, y: u32};
        let mut v = &[F{x: 1, y: 2}, F{x: 3, y: 4}];
        let x = &mut v;
        v[0].y;
        //~^ ERROR cannot use `v[_].y` because it was mutably borrowed
        drop(x);
    }
    // Field of constant index
    {
        struct F {x: u32, y: u32};
        let mut v = &[F{x: 1, y: 2}, F{x: 3, y: 4}];
        let x = &mut v;
        match v {
            &[_, F {x: ref xf, ..}] => println!("{}", xf),
            //~^ ERROR cannot borrow `v[..].x` as immutable because it is also borrowed as mutable
            _ => panic!("other case")
        }
        drop(x);
    }
    // Field from upvar
    {
        let mut x = 0;
        || {
            let y = &mut x;
            &mut x; //~ ERROR cannot borrow `x` as mutable more than once at a time
            *y = 1;
        };
    }
    // Field from upvar nested
    {
        let mut x = 0;
           || {
               || { //~ ERROR captured variable cannot escape `FnMut` closure body
                   let y = &mut x;
                   &mut x; //~ ERROR cannot borrow `x` as mutable more than once at a time
                   *y = 1;
                   drop(y);
                }
           };
    }
    {
        fn foo(x: Vec<i32>) {
            let c = || {
                drop(x);
                drop(x); //~ ERROR use of moved value: `x`
            };
            c();
        }
    }
}
