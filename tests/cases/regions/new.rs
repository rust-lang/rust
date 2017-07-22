pub type A = fn(&bool);

pub type B = for<'a> fn(&'a bool);

pub type C<'a, 'b> = (&'b u8, &'a u16);

pub type D<T: IntoIterator> = T::IntoIter;

pub type E<T: IntoIterator> = T;

pub fn abc(_: &bool) { }

pub fn def(_: bool) { }

pub fn efg(_: &str) { }

pub fn fgh(_: &'static str) { }
