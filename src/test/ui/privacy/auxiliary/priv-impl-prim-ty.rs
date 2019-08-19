pub trait A {
    fn frob(&self);
}

impl A for isize { fn frob(&self) {} }

pub fn frob<T:A>(t: T) {
    t.frob();
}
