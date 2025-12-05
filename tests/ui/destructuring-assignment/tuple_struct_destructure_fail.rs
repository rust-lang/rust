struct TupleStruct<S, T>(S, T);

enum Enum<S, T> {
    SingleVariant(S, T)
}

type Alias<S> = Enum<S, isize>;

trait Test {
    fn test() -> TupleStruct<isize, isize> {
        TupleStruct(0, 0)
    }
}

impl Test for Alias<isize> {}

fn test() -> TupleStruct<isize, isize> {
    TupleStruct(0, 0)
}

fn main() {
    let (mut a, mut b);
    TupleStruct(a, .., b, ..) = TupleStruct(0, 1);
    //~^ ERROR `..` can only be used once per tuple struct or variant pattern
    Enum::SingleVariant(a, .., b, ..) = Enum::SingleVariant(0, 1);
    //~^ ERROR `..` can only be used once per tuple struct or variant pattern

    TupleStruct(a, a, b) = TupleStruct(1, 2);
    //~^ ERROR this pattern has 3 fields, but the corresponding tuple struct has 2 fields
    TupleStruct(_) = TupleStruct(1, 2);
    //~^ ERROR this pattern has 1 field, but the corresponding tuple struct has 2 fields
    Enum::SingleVariant(a, a, b) = Enum::SingleVariant(1, 2);
    //~^ ERROR this pattern has 3 fields, but the corresponding tuple variant has 2 fields
    Enum::SingleVariant(_) = Enum::SingleVariant(1, 2);
    //~^ ERROR this pattern has 1 field, but the corresponding tuple variant has 2 fields

    // Check if `test` is recognized as not a tuple struct but a function call:
    test() = TupleStruct(0, 0);
    //~^ ERROR invalid left-hand side of assignment
    (test)() = TupleStruct(0, 0);
    //~^ ERROR invalid left-hand side of assignment
    <Alias::<isize> as Test>::test() = TupleStruct(0, 0);
    //~^ ERROR invalid left-hand side of assignment
}
