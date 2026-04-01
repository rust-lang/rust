mod b {
    pub struct A(u32);
}

trait Id {
    type Assoc;
}
impl Id for b::A {
    type Assoc = b::A;
}
impl Id for u32 {
    type Assoc = u32;
}


trait Trait<T> {
    fn method(&self)
    where
        T: Id<Assoc = b::A>;
}

impl<T: Id> Trait<T> for <T as Id>::Assoc {
    fn method(&self)
    where
        T: Id<Assoc = b::A>,
    {
        let Self(a) = self;
        //~^ ERROR: tuple struct constructor `A` is private
        println!("{a}");
    }
}

fn main() {}
