#![allow(dead_assignment)]

enum foo<T> { arm(T), }

fn altfoo<T>(f: foo<T>) {
    let mut hit = false;
    match f { foo::arm::<T>(_x) => { println!("in arm"); hit = true; } }
    assert!((hit));
}

pub fn main() { altfoo::<isize>(foo::arm::<isize>(10)); }
