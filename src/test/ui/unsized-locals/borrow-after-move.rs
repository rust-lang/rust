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
        println!("{}", &x);
        //~^ERROR borrow of moved value
        println!("{}", &y);
        //~^ERROR borrow of moved value
    }

    {
        let x = "hello".to_owned().into_boxed_str();
        let y = *x;
        y.foo();
        println!("{}", &x);
        //~^ERROR borrow of moved value
        println!("{}", &y);
        //~^ERROR borrow of moved value
    }

    {
        let x = "hello".to_owned().into_boxed_str();
        x.foo();
        println!("{}", &x);
        //~^ERROR borrow of moved value
    }
}
