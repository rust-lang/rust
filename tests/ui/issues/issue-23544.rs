pub trait A: Copy {}

pub trait D {
    fn f<T>(self)
        where T<Bogus = Self::AlsoBogus>: A;
        //~^ ERROR associated item constraints are not allowed here [E0229]
}

fn main() {}
