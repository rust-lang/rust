//@ check-pass
//@ edition:2021

use std::rc::Rc;

// Test that we restrict precision when moving not-`Copy` types, if any of the parent paths
// implement `Drop`. This is to ensure that we don't move out of a type that implements Drop.
pub fn test1() {
    struct Foo(Rc<i32>);

    impl Drop for Foo {
        fn drop(self: &mut Foo) {}
    }

    let f = Foo(Rc::new(1));
    let x = move || {
        println!("{:?}", f.0);
    };

    x();
}


// Test that we don't restrict precision when moving `Copy` types(i.e. when copying),
// even if any of the parent paths implement `Drop`.
pub fn test2() {
    struct Character {
        hp: u32,
        name: String,
    }

    impl Drop for Character {
        fn drop(&mut self) {}
    }

    let character = Character { hp: 100, name: format!("A") };

    let c = move || {
        println!("{}", character.hp)
    };

    c();

    println!("{}", character.name);
}

fn main() {}
