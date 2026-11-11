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
        drop_unsized(y);
    }

    {
        let x = "hello".to_owned().into_boxed_str();
        let _y = *x; //~ERROR the size for values of type `str` cannot be known at compilation time [E0277]
        drop_unsized(x);
    }

    {
        let x = "hello".to_owned().into_boxed_str();
        drop_unsized(x);
        let _y = *x; //~ERROR the size for values of type `str` cannot be known at compilation time [E0277]
    }

    {
        let x = "hello".to_owned().into_boxed_str();
        let y = *x; //~ERROR the size for values of type `str` cannot be known at compilation time [E0277]
        y.foo();
        y.foo();
    }

    {
        let x = "hello".to_owned().into_boxed_str();
        let _y = *x; //~ERROR the size for values of type `str` cannot be known at compilation time [E0277]
        x.foo();
    }

    {
        let x = "hello".to_owned().into_boxed_str();
        x.foo();
        let _y = *x; //~ERROR the size for values of type `str` cannot be known at compilation time [E0277]
    }
}
