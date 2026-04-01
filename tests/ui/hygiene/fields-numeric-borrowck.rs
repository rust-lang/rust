struct S(u8);

fn main() {
    let mut s = S(0);
    let borrow1 = &mut s.0;
    let S { 0: ref mut borrow2 } = s;
    //~^ ERROR cannot borrow `s.0` as mutable more than once at a time
    borrow2.use_mut();
    borrow1.use_mut();
}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
