// See PR #104857 for details

fn main() {}

fn do_nothing() {}

fn borrow_fn_immut() -> &'static dyn Fn() {
    &do_nothing
}

fn borrow_fn_immut_explicit_return() -> &'static dyn Fn() {
    &do_nothing
}

fn borrow_fn_immut_into_temp() -> &'static dyn Fn() {
    let f = &do_nothing;
    f
}

fn borrow_fn_mut() -> &'static mut dyn FnMut() {
    &mut do_nothing //~ ERROR
}

fn borrow_fn_mut_explicit_return() -> &'static mut dyn FnMut() {
    &mut do_nothing //~ ERROR
}

fn borrow_fn_mut_into_temp() -> &'static mut dyn FnMut() {
    let f = &mut do_nothing;
    f //~ ERROR
}
