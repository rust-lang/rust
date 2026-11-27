// Taken from https://github.com/rust-lang/rust/issues/44454#issuecomment-1175925928

trait Trait<ARG: 'static>: 'static {
    type Assoc: AsRef<str>;
}

fn hr<T: ?Sized, ARG>(x: T::Assoc) -> Box<dyn AsRef<str> + 'static>
where
    T: Trait<ARG>
{
    Box::new(x)
}

fn extend_lt<'a>(x: &'a str) -> Box<dyn AsRef<str> + 'static> {
    type DynTrait = dyn for<'a> Trait<&'a str, Assoc = &'a str>;
    hr::<DynTrait, _>(x) //~ ERROR: borrowed data escapes outside of function
}

fn main() {
    let extended = extend_lt(&String::from("hello"));
    println!("{}", extended.as_ref().as_ref());
}
