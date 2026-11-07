//@ run-pass
#![allow(non_camel_case_types)]

struct closure_box<'a> {
    cl: Box<dyn FnMut() + 'a>,
}

fn box_it<'a>(x: Box<dyn FnMut() + 'a>) -> closure_box<'a> {
    closure_box {cl: x}
}

pub fn main() {
    let mut i = 3;
    assert_eq!(i, 3);
    {
        let cl = || i += 1;
        let mut cl_box = box_it(Box::new(cl));
        (cl_box.cl)();
    }
    assert_eq!(i, 4);
}
