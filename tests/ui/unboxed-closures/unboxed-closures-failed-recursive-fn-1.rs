// Various unsuccessful attempts to put the unboxed closure kind
// inference into an awkward position that might require fixed point
// iteration (basically where inferring the kind of a closure `c`
// would require knowing the kind of `c`). I currently believe this is
// impossible.

fn a() {
    // This case of recursion wouldn't even require fixed-point
    // iteration, but it still doesn't work. The weird structure with
    // the `Option` is to avoid giving any useful hints about the `Fn`
    // kind via the expected type.
    let mut factorial: Option<Box<dyn Fn(u32) -> u32>> = None;

    let f = |x: u32| -> u32 {
        let g = factorial.as_ref().unwrap();
        //~^ ERROR `factorial` does not live long enough
        if x == 0 {1} else {x * g(x-1)}
    };

    factorial = Some(Box::new(f));
    //~^ ERROR cannot assign to `factorial` because it is borrowed
}

fn b() {
    let mut factorial: Option<Box<dyn Fn(u32) -> u32 + 'static>> = None;

    let f = |x: u32| -> u32 {
        let g = factorial.as_ref().unwrap();
        //~^ ERROR `factorial` does not live long enough
        if x == 0 {1} else {x * g(x-1)}
    };

    factorial = Some(Box::new(f));
    //~^ ERROR cannot assign to `factorial` because it is borrowed
}

fn main() { }
