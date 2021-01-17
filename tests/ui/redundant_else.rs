#![warn(clippy::redundant_else)]
#![allow(clippy::needless_return)]

fn main() {
    loop {
        // break
        if foo() {
            println!("Love your neighbor;");
            break;
        } else {
            println!("yet don't pull down your hedge.");
        }
        // continue
        if foo() {
            println!("He that lies down with Dogs,");
            continue;
        } else {
            println!("shall rise up with fleas.");
        }
        // match block
        if foo() {
            match foo() {
                1 => break,
                _ => return,
            }
        } else {
            println!("You may delay, but time will not.");
        }
    }
    // else if
    if foo() {
        return;
    } else if foo() {
        return;
    } else {
        println!("A fat kitchen makes a lean will.");
    }
    // let binding outside of block
    let _ = {
        if foo() {
            return;
        } else {
            1
        }
    };
    // else if with let binding outside of block
    let _ = {
        if foo() {
            return;
        } else if foo() {
            return;
        } else {
            2
        }
    };
    // inside if let
    let _ = if let Some(1) = foo() {
        let _ = 1;
        if foo() {
            return;
        } else {
            1
        }
    } else {
        1
    };

    //
    // non-lint cases
    //

    // sanity check
    if foo() {
        let _ = 1;
    } else {
        println!("Who is wise? He that learns from every one.");
    }
    // else if without else
    if foo() {
        return;
    } else if foo() {
        foo()
    };
    // nested if return
    if foo() {
        if foo() {
            return;
        }
    } else {
        foo()
    };
    // match with non-breaking branch
    if foo() {
        match foo() {
            1 => foo(),
            _ => return,
        }
    } else {
        println!("Three may keep a secret, if two of them are dead.");
    }
    // let binding
    let _ = if foo() {
        return;
    } else {
        1
    };
    // assign
    let a;
    a = if foo() {
        return;
    } else {
        1
    };
    // assign-op
    a += if foo() {
        return;
    } else {
        1
    };
    // if return else if else
    if foo() {
        return;
    } else if foo() {
        1
    } else {
        2
    };
    // if else if return else
    if foo() {
        1
    } else if foo() {
        return;
    } else {
        2
    };
    // else if with let binding
    let _ = if foo() {
        return;
    } else if foo() {
        return;
    } else {
        2
    };
    // inside function call
    Box::new(if foo() {
        return;
    } else {
        1
    });
}

fn foo<T>() -> T {
    unimplemented!("I'm not Santa Claus")
}
