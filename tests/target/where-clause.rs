pub trait Test {
    fn very_long_method_name<F>(self, f: F) -> MyVeryLongReturnType
        where F: FnMut(Self::Item) -> bool;
    fn exactly_100_chars1<F>(self, f: F) -> MyVeryLongReturnType where F: FnMut(Self::Item) -> bool;
}

fn very_long_function_name<F>(very_long_argument: F) -> MyVeryLongReturnType
    where F: FnMut(Self::Item) -> bool
{
}

struct VeryLongTupleStructName<A, B, C, D, E>(LongLongTypename, LongLongTypename, i32, i32)
    where A: LongTrait;

struct Exactly100CharsToSemicolon<A, B, C, D, E>(LongLongTypename, i32, i32) where A: LongTrait1234;

struct AlwaysOnNextLine<LongLongTypename, LongTypename, A, B, C, D, E, F>
    where A: LongTrait
{
    x: i32,
}
