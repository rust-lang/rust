#![feature(structural_match)]
impl<T: ?Sized> std::marker::StructuralPartialEq for O<T> { }

enum O<T: ?Sized> {
    Some(*const T),
    None,
}

const C: O<dyn for<'a> Fn(Box<dyn Fn(&'a u8)>)> = O::None;

fn main() {
    match O::None {
        C => (), //~ ERROR constant of non-structural type
    }
}
