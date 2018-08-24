use std::ops::Deref;

struct Obj<F> where F: FnMut() -> u32 {
    fn_ptr: fn() -> (),
    closure: F,
}

struct C {
    c_fn_ptr: fn() -> (),
}

struct D(C);

impl Deref for D {
    type Target = C;
    fn deref(&self) -> &C {
        &self.0
    }
}


fn empty() {}

fn main() {
    let o = Obj { fn_ptr: empty, closure: || 42 };
    let p = &o;
    p.closure(); //~ ERROR no method named `closure` found
    let q = &p;
    q.fn_ptr(); //~ ERROR no method named `fn_ptr` found
    let r = D(C { c_fn_ptr: empty });
    let s = &r;
    s.c_fn_ptr(); //~ ERROR no method named `c_fn_ptr` found
}
