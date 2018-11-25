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

fn main() {}
