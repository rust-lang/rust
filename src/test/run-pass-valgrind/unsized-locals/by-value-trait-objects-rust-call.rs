#![feature(unsized_locals)]
#![feature(unboxed_closures)]

pub trait FnOnce<Args> {
    type Output;
    extern "rust-call" fn call_once(self, args: Args) -> Self::Output;
}

struct A;

impl FnOnce<()> for A {
    type Output = String;
    extern "rust-call" fn call_once(self, (): ()) -> Self::Output {
        format!("hello")
    }
}

struct B(i32);

impl FnOnce<()> for B {
    type Output = String;
    extern "rust-call" fn call_once(self, (): ()) -> Self::Output {
        format!("{}", self.0)
    }
}

struct C(String);

impl FnOnce<()> for C {
    type Output = String;
    extern "rust-call" fn call_once(self, (): ()) -> Self::Output {
        self.0
    }
}

struct D(Box<String>);

impl FnOnce<()> for D {
    type Output = String;
    extern "rust-call" fn call_once(self, (): ()) -> Self::Output {
        *self.0
    }
}


fn main() {
    let x = *(Box::new(A) as Box<dyn FnOnce<(), Output = String>>);
    assert_eq!(x.call_once(()), format!("hello"));
    let x = *(Box::new(B(42)) as Box<dyn FnOnce<(), Output = String>>);
    assert_eq!(x.call_once(()), format!("42"));
    let x = *(Box::new(C(format!("jumping fox"))) as Box<dyn FnOnce<(), Output = String>>);
    assert_eq!(x.call_once(()), format!("jumping fox"));
    let x = *(Box::new(D(Box::new(format!("lazy dog")))) as Box<dyn FnOnce<(), Output = String>>);
    assert_eq!(x.call_once(()), format!("lazy dog"));
}
