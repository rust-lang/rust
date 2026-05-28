//@ check-pass

const fn foo((Ok(a) | Err(a)): Result<i32, i32>) {
    let x = Ok(3);
    let (Ok(y) | Err(y)) = x;
}

const X: () = {
    let x = Ok(3);
    let (Ok(y) | Err(y)) = x;
};

static Y: () = {
    let x = Ok(3);
    let (Ok(y) | Err(y)) = x;
};

static mut Z: () = {
    let x = Ok(3);
    let (Ok(y) | Err(y)) = x;
};

fn main() {
    let _: [(); {
        let x = Ok(3);
        let (Ok(y) | Err(y)) = x;
        2
    }];
}
