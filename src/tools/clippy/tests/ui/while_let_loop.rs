#![warn(clippy::while_let_loop)]
#![allow(clippy::uninlined_format_args)]
//@no-rustfix
fn main() {
    let y = Some(true);
    loop {
        //~^ ERROR: this loop could be written as a `while let` loop
        //~| NOTE: `-D clippy::while-let-loop` implied by `-D warnings`
        if let Some(_x) = y {
            let _v = 1;
        } else {
            break;
        }
    }

    #[allow(clippy::never_loop)]
    loop {
        // no error, break is not in else clause
        if let Some(_x) = y {
            let _v = 1;
        }
        break;
    }

    loop {
        //~^ ERROR: this loop could be written as a `while let` loop
        match y {
            Some(_x) => true,
            None => break,
        };
    }

    loop {
        //~^ ERROR: this loop could be written as a `while let` loop
        let x = match y {
            Some(x) => x,
            None => break,
        };
        let _x = x;
        let _str = "foo";
    }

    loop {
        //~^ ERROR: this loop could be written as a `while let` loop
        let x = match y {
            Some(x) => x,
            None => break,
        };
        {
            let _a = "bar";
        };
        {
            let _b = "foobar";
        }
    }

    loop {
        // no error, else branch does something other than break
        match y {
            Some(_x) => true,
            _ => {
                let _z = 1;
                break;
            },
        };
    }

    while let Some(x) = y {
        // no error, obviously
        println!("{}", x);
    }

    // #675, this used to have a wrong suggestion
    loop {
        //~^ ERROR: this loop could be written as a `while let` loop
        let (e, l) = match "".split_whitespace().next() {
            Some(word) => (word.is_empty(), word.len()),
            None => break,
        };

        let _ = (e, l);
    }
}

fn issue771() {
    let mut a = 100;
    let b = Some(true);
    loop {
        if a > 10 {
            break;
        }

        match b {
            Some(_) => a = 0,
            None => break,
        }
    }
}

fn issue1017() {
    let r: Result<u32, u32> = Ok(42);
    let mut len = 1337;

    loop {
        match r {
            Err(_) => len = 0,
            Ok(length) => {
                len = length;
                break;
            },
        }
    }
}

#[allow(clippy::never_loop)]
fn issue1948() {
    // should not trigger clippy::while_let_loop lint because break passes an expression
    let a = Some(10);
    let b = loop {
        if let Some(c) = a {
            break Some(c);
        } else {
            break None;
        }
    };
}

fn issue_7913(m: &std::sync::Mutex<Vec<u32>>) {
    // Don't lint. The lock shouldn't be held while printing.
    loop {
        let x = if let Some(x) = m.lock().unwrap().pop() {
            x
        } else {
            break;
        };

        println!("{}", x);
    }
}

fn issue_5715(mut m: core::cell::RefCell<Option<u32>>) {
    // Don't lint. The temporary from `borrow_mut` must be dropped before overwriting the `RefCell`.
    loop {
        let x = if let &mut Some(x) = &mut *m.borrow_mut() {
            x
        } else {
            break;
        };

        m = core::cell::RefCell::new(Some(x + 1));
    }
}
