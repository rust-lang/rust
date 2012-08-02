fn ignore<T>(_t: T) {}

fn nested() {
    let y = 3;
    ignore(fn&(z: &z/int) -> &z/int {
        if false { return &y; } //~ ERROR illegal borrow
        return z;
    });
}

fn main() {}