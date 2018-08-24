pub trait A: Copy {}

struct Foo;

pub trait D {
    fn f<T>(self)
        where T<Bogus = Foo>: A;
        //~^ ERROR associated type bindings are not allowed here [E0229]
}

fn main() {}
