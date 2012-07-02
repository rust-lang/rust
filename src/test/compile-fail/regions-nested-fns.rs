fn ignore<T>(t: T) {}

fn nested(x: &x.int) {
    let y = 3;
    let mut ay = &y;

    ignore(fn&(z: &z.int) {
        ay = x;
        ay = &y;
        ay = z; //~ ERROR references with lifetime
    });

    ignore(fn&(z: &z.int) -> &z.int {
        if false { ret x; }  //~ ERROR references with lifetime
        if false { ret &y; } //~ ERROR references with lifetime
        if false { ret ay; } //~ ERROR references with lifetime
        ret z;
    });
}

fn main() {}