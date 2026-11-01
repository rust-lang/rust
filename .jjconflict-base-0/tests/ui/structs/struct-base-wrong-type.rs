// Check that `base` in `Fru { field: expr, ..base }` must have right type.

struct Foo { a: isize, b: isize }
struct Bar { x: isize }

static bar: Bar = Bar { x: 5 };
static foo: Foo = Foo { a: 2, ..bar }; //~  ERROR mismatched types
static foo_i: Foo = Foo { a: 2, ..4 }; //~  ERROR mismatched types

fn main() {
    let b = Bar { x: 5 };
    let f = Foo { a: 2, ..b };        //~ ERROR mismatched types
    let f__isize = Foo { a: 2, ..4 }; //~ ERROR mismatched types
}
