#![crate_name="a"]
#![crate_type = "lib"]

pub trait i<T>
{
    fn dummy(&self, t: T) -> T { panic!() }
}

pub fn f<T>() -> Box<i<T>+'static> {
    impl<T> i<T> for () { }

    Box::new(()) as Box<i<T>+'static>
}
