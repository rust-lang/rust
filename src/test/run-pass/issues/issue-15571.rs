// run-pass
#![feature(box_syntax)]

fn match_on_local() {
    let mut foo: Option<Box<_>> = Some(box 5);
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
    let mut foo: Option<Box<_>> = Some(box 8);
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
    match_on_arg(Some(box 6));
    match_on_binding();
    match_on_upvar();
}
