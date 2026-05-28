// This issue tests fn_traits overloading on arity.
//@ run-pass

#![feature(fn_traits)]
#![feature(unboxed_closures)]

struct Foo;

impl Fn<(isize, isize)> for Foo {
    extern "rust-call" fn call(&self, args: (isize, isize)) -> Self::Output {
        println!("{:?}", args);
        (args.0 + 1, args.1 + 1)
    }
}

impl FnMut<(isize, isize)> for Foo {
    extern "rust-call" fn call_mut(&mut self, args: (isize, isize)) -> Self::Output {
        println!("{:?}", args);
        (args.0 + 1, args.1 + 1)
    }
}

impl FnOnce<(isize, isize)> for Foo {
    type Output = (isize, isize);
    extern "rust-call" fn call_once(self, args: (isize, isize)) -> Self::Output {
        println!("{:?}", args);
        (args.0 + 1, args.1 + 1)
    }
}

impl Fn<(isize, isize, isize)> for Foo {
    extern "rust-call" fn call(&self, args: (isize, isize, isize)) -> Self::Output {
        println!("{:?}", args);
        (args.0 + 3, args.1 + 3, args.2 + 3)
    }
}

impl FnMut<(isize, isize, isize)> for Foo {
    extern "rust-call" fn call_mut(&mut self, args: (isize, isize, isize)) -> Self::Output {
        println!("{:?}", args);
        (args.0 + 3, args.1 + 3, args.2 + 3)
    }
}
impl FnOnce<(isize, isize, isize)> for Foo {
    type Output = (isize, isize, isize);
    extern "rust-call" fn call_once(self, args: (isize, isize, isize)) -> Self::Output {
        println!("{:?}", args);
        (args.0 + 3, args.1 + 3, args.2 + 3)
    }
}

fn main() {
    let foo = Foo;
    assert_eq!(foo(1, 1), (2, 2));
    assert_eq!(foo(1, 1, 1), (4, 4, 4));
}

// https://github.com/rust-lang/rust/issues/18952
