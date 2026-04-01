//@ known-bug: #119786
//@ edition:2021

fn enum_upvar() {
    type T = impl Copy;
    let foo: T = Some((1u32, 2u32));
    let x = move || {
        match foo {
            None => (),
            Some(yield) => (),
        }
    };
}

pub fn main() {}
