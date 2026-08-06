//@ check-pass

#![warn(unused_variables)]

// A binding initialized by a diverging expression should not be reported as unused.

fn diverge() -> ! {
    loop {}
}

fn test1() {
    let res: Result<u32, u32> = diverge();
    eprintln!("{:?}", res);
}

fn foo() -> ! {
    todo!()
}

fn test2() {
    let res: Result<u32, u32> = foo();
    match res {
        Ok(v) => todo!(),
        Err(err) => todo!(),
    }
}

fn test3() -> i32 {
    let x = {
        //~^ WARN unused variable: `x`
        return 5;
    };
}

fn main() {
    test1();
    test2();
    test3();
}
