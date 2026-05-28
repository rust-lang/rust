// Various unsuccessful attempts to put the unboxed closure kind
// inference into an awkward position that might require fixed point
// iteration (basically where inferring the kind of a closure `c`
// would require knowing the kind of `c`). I currently believe this is
// impossible.

fn a() {
    let mut closure0 = None;
    //~^ ERROR type annotations needed
    let vec = vec![1, 2, 3];

    loop {
        {
            let closure1 = || {
                match closure0.take() {
                    Some(c) => {
                        return c();
                    }
                    None => { }
                }
            };
            closure1();
        }

        closure0 = || vec;
    }
}

fn main() { }
