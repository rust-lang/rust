const _: i32 = {
    let mut x = 0;

    while x < 4 {
        //~^ ERROR constant contains unimplemented expression type
        //~| ERROR constant contains unimplemented expression type
        x += 1;
    }

    while x < 8 {
        x += 1;
    }

    x
};

const _: i32 = {
    let mut x = 0;

    for i in 0..4 {
        //~^ ERROR constant contains unimplemented expression type
        //~| ERROR constant contains unimplemented expression type
        //~| ERROR references in constants may only refer to immutable values
        //~| ERROR calls in constants are limited to constant functions, tuple
        //         structs and tuple variants
        x += i;
    }

    for i in 0..4 {
        x += i;
    }

    x
};

const _: i32 = {
    let mut x = 0;

    loop {
        x += 1;
        if x == 4 {
            //~^ ERROR constant contains unimplemented expression type
            //~| ERROR constant contains unimplemented expression type
            break;
        }
    }

    loop {
        x += 1;
        if x == 8 {
            break;
        }
    }

    x
};

fn main() {}
