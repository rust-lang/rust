// Try to initialise a DST struct where the lost information is deeply nested.
// This is an error because it requires an unsized rvalue. This is a problem
// because it would require stack allocation of an unsized temporary (*g in the
// test).

#![feature(unsized_tuple_coercion)]

pub fn main() {
    let f: ([isize; 3],) = ([5, 6, 7],);
    let g: &([isize],) = &f;
    let h: &(([isize],),) = &(*g,);
    //~^ ERROR the size for values of type
}
