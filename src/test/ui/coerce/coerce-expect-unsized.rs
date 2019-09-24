// run-pass
#![feature(box_syntax)]

use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

// Check that coercions apply at the pointer level and don't cause
// rvalue expressions to be unsized. See #20169 for more information.

pub fn main() {
    let _: Box<[isize]> = Box::new({ [1, 2, 3] });
    let _: Box<[isize]> = Box::new(if true { [1, 2, 3] } else { [1, 3, 4] });
    let _: Box<[isize]> = Box::new(match true { true => [1, 2, 3], false => [1, 3, 4] });
    let _: Box<dyn Fn(isize) -> _> = Box::new({ |x| (x as u8) });
    let _: Box<dyn Debug> = Box::new(if true { false } else { true });
    let _: Box<dyn Debug> = Box::new(match true { true => 'a', false => 'b' });

    let _: &[isize] = &{ [1, 2, 3] };
    let _: &[isize] = &if true { [1, 2, 3] } else { [1, 3, 4] };
    let _: &[isize] = &match true { true => [1, 2, 3], false => [1, 3, 4] };
    let _: &dyn Fn(isize) -> _ = &{ |x| (x as u8) };
    let _: &dyn Debug = &if true { false } else { true };
    let _: &dyn Debug = &match true { true => 'a', false => 'b' };

    let _: &str = &{ String::new() };
    let _: &str = &if true { String::from("...") } else { 5.to_string() };
    let _: &str = &match true {
        true => format!("{}", false),
        false => ["x", "y"].join("+")
    };

    let _: Box<[isize]> = Box::new([1, 2, 3]);
    let _: Box<dyn Fn(isize) -> _> = Box::new(|x| (x as u8));

    let _: Rc<RefCell<[isize]>> = Rc::new(RefCell::new([1, 2, 3]));
    let _: Rc<RefCell<dyn FnMut(isize) -> _>> = Rc::new(RefCell::new(|x| (x as u8)));

    let _: Vec<Box<dyn Fn(isize) -> _>> = vec![
        Box::new(|x| (x as u8)),
        Box::new(|x| (x as i16 as u8)),
    ];
}
