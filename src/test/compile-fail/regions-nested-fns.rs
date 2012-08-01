fn ignore<T>(t: T) {}

fn nested(x: &x/int) {
    let y = 3;
    let mut ay = &y;

    ignore(fn&(z: &z/int) {
        ay = x;
        ay = &y;
        ay = z; //~ ERROR mismatched types
    });

    ignore(fn&(z: &z/int) -> &z/int {
        if false { ret x; }  //~ ERROR mismatched types
        if false { ret ay; } //~ ERROR mismatched types
        ret z;
    });
}

fn main() {}