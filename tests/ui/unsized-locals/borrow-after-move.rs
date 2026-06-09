#![feature(unsized_fn_params)]

pub trait Foo {
    fn foo(self) -> String;
}

impl Foo for str {
    fn foo(self) -> String {
        self.to_owned()
    }
}

fn drop_unsized<T: ?Sized>(_: T) {}

fn main() {
    {
        let x = "hello".to_owned().into_boxed_str();
        let y = *x; //~ERROR the size for values of type `str` cannot be known at compilation time [E0277]
        drop_unsized(y);
        println!("{}", &x);
        println!("{}", &y);
    }

    {
        let x = "hello".to_owned().into_boxed_str();
        let y = *x; //~ERROR the size for values of type `str` cannot be known at compilation time [E0277]
        y.foo();
        println!("{}", &x);
        println!("{}", &y);
    }

    {
        let x = "hello".to_owned().into_boxed_str();
        x.foo();
        println!("{}", &x);
    }
}
