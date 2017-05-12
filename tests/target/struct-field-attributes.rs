// #1535
#![feature(struct_field_attributes)]

struct Foo {
    bar: u64,

    #[cfg(test)]
    qux: u64,
}

fn do_something() -> Foo {
    Foo {
        bar: 0,

        #[cfg(test)]
        qux: 1,
    }
}

fn main() {
    do_something();
}
