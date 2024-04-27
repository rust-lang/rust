//@ check-pass

macro_rules! make_struct {
    ($name:ident) => {
        #[derive(Debug)]
        struct Foo {
            #[cfg(not(FALSE))]
            field: fn($name: bool)
        }
    }
}

make_struct!(param_name);

fn main() {}
