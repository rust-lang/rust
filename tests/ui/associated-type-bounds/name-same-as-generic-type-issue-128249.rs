trait Trait<Type> {
    type Type;

    fn one(&self, val:  impl Trait<Type: Default>);
    //~^ ERROR trait takes 1 generic argument but 0 generic arguments were supplied

    fn two<T: Trait<Type: Default>>(&self) -> T;
    //~^ ERROR trait takes 1 generic argument but 0 generic arguments were supplied

    fn three<T>(&self) -> T where
        T: Trait<Type: Default>,;
    //~^ ERROR trait takes 1 generic argument but 0 generic arguments were supplied
}

fn main() {}
