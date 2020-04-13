#![feature(destructuring_assignment)]

struct TupleStruct<S, T>(S, T);

fn main() {
    let (mut a, mut b);
    TupleStruct(a, .., b, ..) = TupleStruct(0, 1);
    //~^ ERROR `..` can only be used once per tuple struct pattern
    TupleStruct(a, a, b) = TupleStruct(1,2);
    //~^ ERROR this pattern has 3 fields, but the corresponding tuple struct has 2 fields
    // Check if `test` is recognized as not a tuple struct but a function call:
    test() = TupleStruct(0,0); //~ ERROR invalid left-hand side of assignment
    TupleStruct(_) = TupleStruct(1,2);
    //~^ ERROR this pattern has 1 field, but the corresponding tuple struct has 2 fields
}

fn test() -> TupleStruct<isize, isize> {
    TupleStruct(0, 0)
}
