fn main() {
    let foo = 1;
    let bar = 2;
    let foobar = foo + bar;

    let nope = optint(0) + optint(0);
    let somefoo = optint(foo) + optint(0);
    let somebar = optint(bar) + optint(0);
    let somefoobar = optint(foo) + optint(bar);

    match nope {
        None => (),
        Some(foo) => fail!(fmt!("expected None, but found %?", foo))
    }
    fail_unless!(foo == somefoo.get());
    fail_unless!(bar == somebar.get());
    fail_unless!(foobar == somefoobar.get());
}

fn optint(in: int) -> Option<int> {
    if in == 0 {
        return None;
    }
    else {
        return Some(in);
    }
}
