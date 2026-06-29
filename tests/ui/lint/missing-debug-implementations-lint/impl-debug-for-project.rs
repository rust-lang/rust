//@ check-pass

#![deny(missing_debug_implementations)]

pub struct Local;

impl std::fmt::Debug for <Local as Identity>::Output {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { Ok(()) }
}

pub trait Identity { type Output; }
impl<T> Identity for T { type Output = T; }

pub struct Generic<T>(T);

impl std::fmt::Debug for <Generic<i32> as Identity>::Output {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { Ok(()) }
}

fn main() {}
