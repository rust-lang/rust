struct A<T, 'a> {
    t: &'a T,
}

struct B<T, 'a, U> {
    t: &'a T,
    u: U,
}

struct C<T, U, 'a> {
    t: &'a T,
    u: U,
}

struct D<T, U, 'a, 'b, V, 'c> {
    t: &'a T,
    u: &'b U,
    v: &'c V,
}

fn main() {}
