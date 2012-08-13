fn ignore<T>(t: T) {}

fn nested(x: &x/int) {
    let y = 3;
    let mut ay = &y; //~ ERROR cannot infer an appropriate lifetime

    ignore(fn&(z: &z/int) {
        ay = x;
        ay = &y;  //~ ERROR cannot infer an appropriate lifetime
        ay = z;
    });

    ignore(fn&(z: &z/int) -> &z/int {
        if false { return x; }  //~ ERROR mismatched types
        if false { return ay; }
        return z;
    });
}

fn main() {}