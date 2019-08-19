#![feature(fn_traits)]

struct ClosureBox<'a> {
    cl: Box<dyn FnMut() + 'a>,
}

fn box_it<'r>(x: Box<dyn FnMut() + 'r>) -> ClosureBox<'r> {
    ClosureBox {cl: x}
}

fn main() {
    let mut cl_box = {
        let mut i = 3;
        box_it(Box::new(|| i += 1)) //~ ERROR `i` does not live long enough
    };
    cl_box.cl.call_mut(());
}
