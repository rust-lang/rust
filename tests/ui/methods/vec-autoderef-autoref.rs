//! Test that method resolution does not autoderef `Vec`
//! into a slice or perform additional autorefs.

fn main() {
    let mut a = vec![0];
    a.test_mut(); //~ ERROR no method named `test_mut` found
    a.test(); //~ ERROR no method named `test` found

    ([1]).test(); //~ ERROR no method named `test` found
    (&[1]).test(); //~ ERROR no method named `test` found
}

trait MyIter {
    fn test_mut(&mut self);
    fn test(&self);
}

impl<'a> MyIter for &'a [isize] {
    fn test_mut(&mut self) { }
    fn test(&self) { }
}

impl<'a> MyIter for &'a str {
    fn test_mut(&mut self) { }
    fn test(&self) { }
}
