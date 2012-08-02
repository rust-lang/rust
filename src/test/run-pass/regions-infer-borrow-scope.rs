type point = {x: int, y: int};

fn x_coord(p: &point) -> &int {
    return &p.x;
}

fn main() {
    let p = @{x: 3, y: 4};
    let xc = x_coord(p);
    assert *xc == 3;
}

