struct closure_box {
    cl: &fn();
}

fn box_it(x: &r/fn()) -> closure_box/&r {
    closure_box {cl: x}
}

fn call_static_closure(cl: closure_box/&static) {
    cl.cl();
}

fn main() {
    let cl_box = box_it(|| debug!("Hello, world!"));
    call_static_closure(cl_box);
}
