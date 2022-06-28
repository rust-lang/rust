// check-pass

#[allow(unconditional_recursion)]
fn foo(b: bool) -> impl Copy {
    let (mut x, mut y) = foo(false);
    x = 42;
    y = "foo";
    if b {
        panic!()
    } else {
        foo(true)
    }
}

fn bar(b: bool) -> Option<impl Copy> {
    if b {
        return None;
    }
    match bar(!b) {
        Some((mut x, mut y)) => {
            x = 42;
            y = "foo";
        }
        None => {}
    }
    None
}

fn main() {}
