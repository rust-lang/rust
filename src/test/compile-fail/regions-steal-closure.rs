struct closure_box {
    cl: &fn()
}

fn box_it(x: &r/fn()) -> closure_box/&r {
    closure_box {cl: x}
}

fn main() {
    let cl_box = {
        let mut i = 3;
        box_it(|| i += 1) //~ ERROR cannot infer an appropriate lifetime
    };
    cl_box.cl();
}
