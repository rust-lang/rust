fn ignore<T>(_t: T) {}

fn nested() {
    let y = 3;
    ignore(fn&(z: &z/int) -> &z/int {
        if false { ret &y; } //~ ERROR illegal borrow
        ret z;
    });
}

fn main() {}