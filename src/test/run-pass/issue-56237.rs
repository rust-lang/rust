use std::ops::Deref;

fn foo<P>(_value: <P as Deref>::Target)
where
    P: Deref,
    <P as Deref>::Target: Sized,
{}

fn main() {
    foo::<Box<u32>>(2);
}
