struct Point {
    x: isize,
    y: isize,
}

fn x_coord<'r>(p: &'r Point) -> &'r isize {
    return &p.x;
}

fn foo<'a>(p: Box<Point>) -> &'a isize {
    let xc = x_coord(&*p);
    assert_eq!(*xc, 3);
    return xc; //~ ERROR cannot return value referencing local data `*p`
}

fn main() {}
