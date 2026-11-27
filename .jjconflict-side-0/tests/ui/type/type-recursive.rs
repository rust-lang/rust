struct T1 { //~ ERROR E0072
    foo: isize,
    foolish: T1,
}

struct T2 { //~ ERROR E0072
    inner: Option<T2>,
}

type OptionT3 = Option<T3>;

struct T3 { //~ ERROR E0072
    inner: OptionT3,
}

struct T4(Option<T4>); //~ ERROR E0072

enum T5 { //~ ERROR E0072
    Variant(Option<T5>),
}

enum T6 { //~ ERROR E0072
    Variant{ field: Option<T6> },
}

struct T7 { //~ ERROR E0072
    foo: std::cell::Cell<Option<T7>>,
}

fn main() { }
