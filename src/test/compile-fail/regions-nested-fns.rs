// xfail-test

fn ignore<T>(t: T) {}

fn nested(x: &x.int) {
    let y = 3;
    let mut ay = &y;

    ignore(fn&(z: &z.int) {
        ay = x;
        ay = &y;
        ay = z; //! ERROR foo
    });

    ignore(fn&(z: &z.int) -> &z.int {
        if false { ret x; }  //! ERROR bar
        if false { ret &y; } //! ERROR bar
        if false { ret ay; } //! ERROR bar
        ret z;
    });
}

fn main() {}