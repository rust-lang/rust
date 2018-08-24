struct point {
    x: isize,
    y: isize,
}

fn x_coord<'r>(p: &'r point) -> &'r isize {
    return &p.x;
}

fn foo<'a>(p: Box<point>) -> &'a isize {
    let xc = x_coord(&*p); //~ ERROR `*p` does not live long enough
    assert_eq!(*xc, 3);
    return xc;
}

fn main() {}
