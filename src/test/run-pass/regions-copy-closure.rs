struct closure_box {
    cl: &fn(),
}

fn box_it(+x: &r/fn()) -> closure_box/&r {
    closure_box {cl: move x}
}

fn main() {
    let mut i = 3;
    let cl_box = box_it(|| i += 1);
    assert i == 3;
    (cl_box.cl)();
    assert i == 4;
}
