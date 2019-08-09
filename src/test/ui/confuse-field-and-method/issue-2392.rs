struct FuncContainer {
    f1: fn(data: u8),
    f2: extern "C" fn(data: u8),
    f3: unsafe fn(data: u8),
}

struct FuncContainerOuter {
    container: Box<FuncContainer>
}

struct Obj<F> where F: FnOnce() -> u32 {
    closure: F,
    not_closure: usize,
}

struct BoxedObj {
    boxed_closure: Box<dyn FnOnce() -> u32>,
}

struct Wrapper<F> where F: FnMut() -> u32 {
    wrap: Obj<F>,
}

fn func() -> u32 {
    0
}

fn check_expression() -> Obj<Box<dyn FnOnce() -> u32>> {
    Obj { closure: Box::new(|| 42_u32) as Box<dyn FnOnce() -> u32>, not_closure: 42 }
}

fn main() {
    // test variations of function

    let o_closure = Obj { closure: || 42, not_closure: 42 };
    o_closure.closure(); //~ ERROR no method named `closure` found

    o_closure.not_closure();
    //~^ ERROR no method named `not_closure` found

    let o_func = Obj { closure: func, not_closure: 5 };
    o_func.closure(); //~ ERROR no method named `closure` found

    let boxed_fn = BoxedObj { boxed_closure: Box::new(func) };
    boxed_fn.boxed_closure();//~ ERROR no method named `boxed_closure` found

    let boxed_closure = BoxedObj { boxed_closure: Box::new(|| 42_u32) as Box<dyn FnOnce() -> u32> };
    boxed_closure.boxed_closure();//~ ERROR no method named `boxed_closure` found

    // test expression writing in the notes

    let w = Wrapper { wrap: o_func };
    w.wrap.closure();//~ ERROR no method named `closure` found

    w.wrap.not_closure();
    //~^ ERROR no method named `not_closure` found

    check_expression().closure();//~ ERROR no method named `closure` found
}

impl FuncContainerOuter {
    fn run(&self) {
        unsafe {
            (*self.container).f1(1); //~ ERROR no method named `f1` found
            (*self.container).f2(1); //~ ERROR no method named `f2` found
            (*self.container).f3(1); //~ ERROR no method named `f3` found
        }
    }
}
