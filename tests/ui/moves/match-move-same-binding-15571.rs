//! Regression test for https://github.com/rust-lang/rust/issues/15571

//@ run-pass

fn match_on_local() {
    let mut foo: Option<Box<_>> = Some(Box::new(5));
    match foo {
        None => {},
        Some(x) => {
            foo = Some(x);
        }
    }
    println!("'{}'", foo.unwrap());
}

fn match_on_arg(mut foo: Option<Box<i32>>) {
    match foo {
        None => {}
        Some(x) => {
            foo = Some(x);
        }
    }
    println!("'{}'", foo.unwrap());
}

fn match_on_binding() {
    match Some(Box::new(7)) {
        mut foo => {
            match foo {
                None => {},
                Some(x) => {
                    foo = Some(x);
                }
            }
            println!("'{}'", foo.unwrap());
        }
    }
}

fn match_on_upvar() {
    let mut foo: Option<Box<_>> = Some(Box::new(8));
    let f = move|| {
        match foo {
            None => {},
            Some(x) => {
                foo = Some(x);
            }
        }
        println!("'{}'", foo.unwrap());
    };
    f();
}

fn main() {
    match_on_local();
    match_on_arg(Some(Box::new(6)));
    match_on_binding();
    match_on_upvar();
}
