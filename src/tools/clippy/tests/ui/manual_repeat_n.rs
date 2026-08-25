#![warn(clippy::manual_repeat_n)]

use std::iter::repeat;

fn main() {
    let _ = repeat(10).take(3);
    //~^ manual_repeat_n

    let _ = repeat(String::from("foo")).take(4);
    //~^ manual_repeat_n

    for value in std::iter::repeat(5).take(3) {}
    //~^ manual_repeat_n

    let _: Vec<_> = std::iter::repeat(String::from("bar")).take(10).collect();
    //~^ manual_repeat_n

    let _ = repeat(vec![1, 2]).take(2);
    //~^ manual_repeat_n
}

mod foo_lib {
    pub fn iter() -> std::iter::Take<std::iter::Repeat<&'static [u8]>> {
        todo!()
    }
}

fn foo() {
    let _ = match 1 {
        1 => foo_lib::iter(),
        // Shouldn't lint because `external_lib::iter` doesn't return `std::iter::RepeatN`.
        2 => std::iter::repeat([1, 2].as_slice()).take(2),
        _ => todo!(),
    };
}
