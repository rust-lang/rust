type point = {x: int, y: int};

fn x_coord(p: &point) -> &int {
    ret &p.x;
}

fn foo(p: @point) -> &int {
    let xc = x_coord(p);
    assert *xc == 3;
    ret xc; //~ ERROR mismatched types: expected `&int` but found
}

fn main() {}

