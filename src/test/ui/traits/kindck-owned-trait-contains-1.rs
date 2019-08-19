// run-pass
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

#![feature(box_syntax)]

trait repeat<A> { fn get(&self) -> A; }

impl<A:Clone + 'static> repeat<A> for Box<A> {
    fn get(&self) -> A {
        (**self).clone()
    }
}

fn repeater<A:Clone + 'static>(v: Box<A>) -> Box<dyn repeat<A>+'static> {
    box v as Box<dyn repeat<A>+'static> // No
}

pub fn main() {
    let x = 3;
    let y = repeater(box x);
    assert_eq!(x, y.get());
}
