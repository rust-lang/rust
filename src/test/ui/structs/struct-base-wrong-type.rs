// Check that `base` in `Fru { field: expr, ..base }` must have right type.
//
// See also struct-base-wrong-type-2.rs, which tests same condition
// within a function body.

struct Foo { a: isize, b: isize }
struct Bar { x: isize }

static bar: Bar = Bar { x: 5 };
static foo: Foo = Foo { a: 2, ..bar }; //~  ERROR mismatched types
                                       //~| expected type `Foo`
                                       //~| found type `Bar`
                                       //~| expected struct `Foo`, found struct `Bar`
static foo_i: Foo = Foo { a: 2, ..4 }; //~  ERROR mismatched types
                                       //~| expected type `Foo`
                                       //~| found type `{integer}`
                                       //~| expected struct `Foo`, found integer

fn main() {
    let b = Bar { x: 5 };
    // See also struct-base-wrong-type-2.rs, which checks these errors on isolation.
    let f = Foo { a: 2, ..b };        //~ ERROR mismatched types
    let f__isize = Foo { a: 2, ..4 }; //~ ERROR mismatched types
}
