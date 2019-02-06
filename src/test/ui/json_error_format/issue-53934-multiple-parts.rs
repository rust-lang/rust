// compile-flags: --error-format pretty-json -Zunstable-options

struct Point { x: isize, y: isize }

fn main() {
    let p = Point { x: 1, y: 2 };
    let Point { .., y, } = p;
}
