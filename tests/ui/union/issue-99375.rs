//@ check-pass

union URes<R: Copy> {
    uninit: (),
    init: R,
}

struct Params<F, R: Copy> {
    function: F,
    result: URes<R>,
}

unsafe extern "C" fn do_call<F, R>(params: *mut Params<F, R>)
where
    R: Copy,
    F: Fn() -> R,
{
    (*params).result.init = ((*params).function)();
}

fn main() {}
