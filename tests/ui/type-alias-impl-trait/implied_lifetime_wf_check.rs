#![feature(type_alias_impl_trait)]

//@ revisions: pass error

//@[pass] check-pass
//@[error] check-fail

type Alias = impl Sized;

#[define_opaque(Alias)]
fn constrain() -> Alias {
    1i32
}

trait HideIt {
    type Assoc;
}

impl HideIt for () {
    type Assoc = Alias;
}

pub trait Yay {}

impl Yay for <() as HideIt>::Assoc {}
#[cfg(error)]
impl Yay for i32 {}
//[error]~^ error conflicting implementations

fn main() {}
