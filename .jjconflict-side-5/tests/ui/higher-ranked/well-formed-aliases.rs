trait Trait {
    type Gat<U: ?Sized>;
}

fn test<T>(f: for<'a> fn(<&'a T as Trait>::Gat<&'a [str]>)) where for<'a> &'a T: Trait {}
//~^ ERROR the size for values of type `str` cannot be known at compilation time

fn main() {}
