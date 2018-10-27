#![feature(unsized_locals)]

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
        let y = *x;
        drop_unsized(y);
        drop_unsized(y); //~ERROR use of moved value
    }

    {
        let x = "hello".to_owned().into_boxed_str();
        let _y = *x;
        drop_unsized(x); //~ERROR use of moved value
    }

    {
        let x = "hello".to_owned().into_boxed_str();
        drop_unsized(x);
        let _y = *x; //~ERROR use of moved value
    }

    {
        let x = "hello".to_owned().into_boxed_str();
        let y = *x;
        y.foo();
        y.foo(); //~ERROR use of moved value
    }

    {
        let x = "hello".to_owned().into_boxed_str();
        let _y = *x;
        x.foo(); //~ERROR use of moved value
    }

    {
        let x = "hello".to_owned().into_boxed_str();
        x.foo();
        let _y = *x; //~ERROR use of moved value
    }
}
