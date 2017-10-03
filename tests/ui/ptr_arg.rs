

#![allow(unused, many_single_char_names)]
#![warn(ptr_arg)]

fn do_vec(x: &Vec<i64>) {
    //Nothing here
}

fn do_vec_mut(x: &mut Vec<i64>) { // no error here
    //Nothing here
}

fn do_str(x: &String) {
    //Nothing here either
}

fn do_str_mut(x: &mut String) { // no error here
    //Nothing here either
}

fn main() {
}

trait Foo {
    type Item;
    fn do_vec(x: &Vec<i64>);
    fn do_item(x: &Self::Item);
}

struct Bar;

// no error, in trait impl (#425)
impl Foo for Bar {
    type Item = Vec<u8>;
    fn do_vec(x: &Vec<i64>) {}
    fn do_item(x: &Vec<u8>) {}
}

fn cloned(x: &Vec<u8>) -> Vec<u8> {
    let e = x.clone();
    let f = e.clone(); // OK
    let g = x;
    let h = g.clone(); // Alas, we cannot reliably detect this without following data.
    let i = (e).clone();
    x.clone()
}

fn str_cloned(x: &String) -> String {
    let a = x.clone();
    let b = x.clone();
    let c = b.clone();
    let d = a.clone()
             .clone()
             .clone();
    x.clone()
}

fn false_positive_capacity(x: &Vec<u8>, y: &String) {
    let a = x.capacity();
    let b = y.clone();
    let c = y.as_str();
}

fn false_positive_capacity_too(x: &String) -> String {
    if x.capacity() > 1024 { panic!("Too large!"); }
    x.clone()
}

