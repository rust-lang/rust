#![feature(or_patterns)]

const fn foo((Ok(a) | Err(a)): Result<i32, i32>) {
    //~^ ERROR or-pattern is not allowed in a `const fn`
    let x = Ok(3);
    let Ok(y) | Err(y) = x;
    //~^ ERROR or-pattern is not allowed in a `const fn`
}

const X: () = {
    let x = Ok(3);
    let Ok(y) | Err(y) = x;
    //~^ ERROR or-pattern is not allowed in a `const`
};

static Y: () = {
    let x = Ok(3);
    let Ok(y) | Err(y) = x;
    //~^ ERROR or-pattern is not allowed in a `static`
};

static mut Z: () = {
    let x = Ok(3);
    let Ok(y) | Err(y) = x;
    //~^ ERROR or-pattern is not allowed in a `static mut`
};

fn main() {
    let _: [(); {
        let x = Ok(3);
        let Ok(y) | Err(y) = x;
        //~^ ERROR or-pattern is not allowed in a `const`
        2
    }];
}
