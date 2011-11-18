

fn id<copy T>(t: T) -> T { ret t; }

fn main() {
    let t = {a: 1, b: 2, c: 3, d: 4, e: 5, f: 6, g: 7};
    assert (t.f == 6);
    let f0 = bind id(t);
    assert (f0().f == 6);
}
