//@ check-pass

#![feature(associated_type_defaults)]

trait Tr {
    type Item = u8;
    type Container = Vec<Self::Item>;
}

impl Tr for () {}

impl Tr for u16 {
    type Item = u16;
}

impl Tr for String {
    type Container = String;
}

impl Tr for usize {
    type Item = u32;
    type Container = Vec<()>;
}

fn main() {
    let _container: <() as Tr>::Container = Vec::<u8>::new();
    let _item: <() as Tr>::Item = 0u8;

    let _container: <u16 as Tr>::Container = Vec::<u16>::new();
    let _item: <u16 as Tr>::Item = 0u16;

    let _container: <String as Tr>::Container = String::new();
    let _item: <String as Tr>::Item = 0u8;

    let _container: <usize as Tr>::Container = Vec::<()>::new();
    let _item: <usize as Tr>::Item = 0u32;
}
