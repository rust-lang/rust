//@ check-pass
//@ edition: 2021

fn foo<'r#struct>() {}

fn hr<T>() where for<'r#struct> T: Into<&'r#struct ()> {}

trait Foo<'r#struct> {}

trait Bar<'r#struct> {
    fn method(&'r#struct self) {}
    fn method2(self: &'r#struct Self) {}
}

fn labeled() {
    'r#struct: loop {
        break 'r#struct;
    }
}

fn main() {}
