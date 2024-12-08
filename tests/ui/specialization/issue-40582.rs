//@ check-pass
//@ known-bug: #40582

// Should fail. Should not be possible to implement `make_static`.

#![feature(specialization)]
#![allow(incomplete_features)]

trait FromRef<'a, T: ?Sized> {
    fn from_ref(r: &'a T) -> Self;
}

impl<'a, T: ?Sized> FromRef<'a, T> for &'a T {
    fn from_ref(r: &'a T) -> Self {
        r
    }
}

impl<'a, T: ?Sized, R> FromRef<'a, T> for R {
    default fn from_ref(_: &'a T) -> Self {
        unimplemented!()
    }
}

fn make_static<T: ?Sized>(data: &T) -> &'static T {
    fn helper<T: ?Sized, R>(data: &T) -> R {
        R::from_ref(data)
    }
    helper(data)
}

fn main() {
    let s = "specialization".to_owned();
    println!("{:?}", make_static(s.as_str()));
}
