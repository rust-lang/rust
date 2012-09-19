// Tests that type assignability is used to search for instances when
// making method calls, but only if there aren't any matches without
// it.

trait iterable<A> {
    fn iterate(blk: fn(x: &A) -> bool);
}

impl<A> &[A]: iterable<A> {
    fn iterate(f: fn(x: &A) -> bool) {
        for vec::each(self) |e| {
            if !f(e) { break; }
        }
    }
}

impl<A> ~[A]: iterable<A> {
    fn iterate(f: fn(x: &A) -> bool) {
        for vec::each(self) |e| {
            if !f(e) { break; }
        }
    }
}

fn length<A, T: iterable<A>>(x: T) -> uint {
    let mut len = 0;
    for x.iterate() |_y| { len += 1 }
    return len;
}

fn main() {
    let x = ~[0,1,2,3];
    // Call a method
    for x.iterate() |y| { assert x[*y] == *y; }
    // Call a parameterized function
    assert length(x) == vec::len(x);
    // Call a parameterized function, with type arguments that require
    // a borrow
    assert length::<int, &[int]>(x) == vec::len(x);

    // Now try it with a type that *needs* to be borrowed
    let z = [0,1,2,3]/_;
    // Call a method
    for z.iterate() |y| { assert z[*y] == *y; }
    // Call a parameterized function
    assert length::<int, &[int]>(z) == vec::len(z);
}
