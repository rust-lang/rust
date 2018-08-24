// Check that `base` in `Fru { field: expr, ..base }` must have right type.
//
// See also struct-base-wrong-type.rs, which tests same condition
// within a const expression.

struct Foo { a: isize, b: isize }
struct Bar { x: isize }

fn main() {
    let b = Bar { x: 5 };
    let f = Foo { a: 2, ..b }; //~  ERROR mismatched types
                               //~| expected type `Foo`
                               //~| found type `Bar`
                               //~| expected struct `Foo`, found struct `Bar`
    let f__isize = Foo { a: 2, ..4 }; //~  ERROR mismatched types
                                 //~| expected type `Foo`
                                 //~| found type `{integer}`
                                 //~| expected struct `Foo`, found integral variable
}
