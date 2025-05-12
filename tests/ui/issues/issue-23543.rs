pub trait A: Copy {}

struct Foo;

pub trait D {
    fn f<T>(self)
        where T<Bogus = Foo>: A;
        //~^ ERROR associated item constraints are not allowed here [E0229]
}

fn main() {}
