struct A<T, 'a> { //~ ERROR incorrect parameter order
    t: &'a T,
}

struct B<T, 'a, U> { //~ ERROR incorrect parameter order
    t: &'a T,
    u: U,
}

struct C<T, U, 'a> { //~ ERROR incorrect parameter order
    t: &'a T,
    u: U,
}

struct D<T, U, 'a, 'b, V, 'c> { //~ ERROR incorrect parameter order
    t: &'a T,
    u: &'b U,
    v: &'c V,
}

fn main() {}
