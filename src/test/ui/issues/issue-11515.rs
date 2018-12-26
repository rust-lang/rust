#![feature(box_syntax)]

struct Test {
    func: Box<FnMut()+'static>
}

fn main() {
    let closure: Box<Fn()+'static> = Box::new(|| ());
    let test = box Test { func: closure }; //~ ERROR mismatched types
}
