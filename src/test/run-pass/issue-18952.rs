// This issue tests fn_traits overloading on arity.

#![feature(fn_traits)]
#![feature(unboxed_closures)]

struct Foo;

impl Fn<(isize, isize)> for Foo {
    extern "rust-call" fn call(&self, args: (isize, isize)) -> Self::Output {
        println!("{:?}", args);
    }
}

impl FnMut<(isize, isize)> for Foo {
    extern "rust-call" fn call_mut(&mut self, args: (isize, isize)) -> Self::Output {
        println!("{:?}", args);
    }
}

impl FnOnce<(isize, isize)> for Foo {
    type Output = ();
    extern "rust-call" fn call_once(self, args: (isize, isize)) -> Self::Output {
        println!("{:?}", args);
    }
}

impl Fn<(isize, isize, isize)> for Foo {
    extern "rust-call" fn call(&self, args: (isize, isize, isize)) -> Self::Output {
        println!("{:?}", args);
    }
}

impl FnMut<(isize, isize, isize)> for Foo {
    extern "rust-call" fn call_mut(&mut self, args: (isize, isize, isize)) -> Self::Output {
        println!("{:?}", args);
    }
}
impl FnOnce<(isize, isize, isize)> for Foo {
    type Output = ();
    extern "rust-call" fn call_once(self, args: (isize, isize, isize)) -> Self::Output {
        println!("{:?}", args);
    }
}

fn main() {
    let foo = Foo;
    foo(1, 1);
}
