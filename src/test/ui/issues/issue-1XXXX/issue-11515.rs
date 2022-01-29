#![feature(box_syntax)]

struct Test {
    func: Box<dyn FnMut() + 'static>,
}

fn main() {
    let closure: Box<dyn Fn() + 'static> = Box::new(|| ());
    let test = box Test { func: closure }; //~ ERROR trait upcasting coercion is experimental [E0658]
}
