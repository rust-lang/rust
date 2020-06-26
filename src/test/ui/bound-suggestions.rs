// run-rustfix

#[allow(dead_code)]
fn test_impl(t: impl Sized) {
    println!("{:?}", t);
    //~^ ERROR doesn't implement
}

#[allow(dead_code)]
fn test_no_bounds<T>(t: T) {
    println!("{:?}", t);
    //~^ ERROR doesn't implement
}

#[allow(dead_code)]
fn test_one_bound<T: Sized>(t: T) {
    println!("{:?}", t);
    //~^ ERROR doesn't implement
}

#[allow(dead_code)]
fn test_no_bounds_where<X, Y>(x: X, y: Y) where X: std::fmt::Debug, {
    println!("{:?} {:?}", x, y);
    //~^ ERROR doesn't implement
}

#[allow(dead_code)]
fn test_one_bound_where<X>(x: X) where X: Sized {
    println!("{:?}", x);
    //~^ ERROR doesn't implement
}

#[allow(dead_code)]
fn test_many_bounds_where<X>(x: X) where X: Sized, X: Sized {
    println!("{:?}", x);
    //~^ ERROR doesn't implement
}

pub fn main() { }
