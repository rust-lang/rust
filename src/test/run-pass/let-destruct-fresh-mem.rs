fn main() {
    let u = {x: 10, y: @{a: 20}};
    let {x, y: @{a}} = u;
    x = 100;
    a = 100;
    assert x == 100;
    assert a == 100;
    assert u.x == 10;
    assert u.y.a == 20;
}
