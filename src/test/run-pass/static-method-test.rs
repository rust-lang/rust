
// A trait for objects that can be used to do an if-then-else
// (No actual need for this to be static, but it is a simple test.)
trait bool_like {
    static fn select<A>(b: self, +x1: A, +x2: A) -> A;
}

fn andand<T: bool_like copy>(x1: T, x2: T) -> T {
    select(x1, x2, x1)
}

impl bool: bool_like {
    static fn select<A>(&&b: bool, +x1: A, +x2: A) -> A {
        if b { x1 } else { x2 }
    }
}

impl int: bool_like {
    static fn select<A>(&&b: int, +x1: A, +x2: A) -> A {
        if b != 0 { x1 } else { x2 }
    }
}

// A trait for sequences that can be constructed imperatively.
trait buildable<A> {
     static pure fn build_sized(size: uint,
                                builder: fn(push: pure fn(+A))) -> self;
}


impl<A> @[A]: buildable<A> {
    #[inline(always)]
     static pure fn build_sized(size: uint,
                                builder: fn(push: pure fn(+A))) -> @[A] {
         at_vec::build_sized(size, builder)
     }
}
impl<A> ~[A]: buildable<A> {
    #[inline(always)]
     static pure fn build_sized(size: uint,
                                builder: fn(push: pure fn(+A))) -> ~[A] {
         vec::build_sized(size, builder)
     }
}

#[inline(always)]
pure fn build<A, B: buildable<A>>(builder: fn(push: pure fn(+A))) -> B {
    build_sized(4, builder)
}

/// Apply a function to each element of an iterable and return the results
fn map<T, IT: BaseIter<T>, U, BU: buildable<U>>
    (v: IT, f: fn(T) -> U) -> BU {
    do build |push| {
        for v.each() |elem| {
            push(f(elem));
        }
    }
}

fn seq_range<BT: buildable<int>>(lo: uint, hi: uint) -> BT {
    do build_sized(hi-lo) |push| {
        for uint::range(lo, hi) |i| {
            push(i as int);
        }
    }
}

fn main() {
    assert seq_range(0, 10) == @[0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

    assert map(&[1,2,3], |x| 1+x) == @[2, 3, 4];
    assert map(&[1,2,3], |x| 1+x) == ~[2, 3, 4];

    assert select(true, 9, 14) == 9;
    assert !andand(true, false);
    assert andand(7, 12) == 12;
    assert andand(0, 12) == 0;
}
