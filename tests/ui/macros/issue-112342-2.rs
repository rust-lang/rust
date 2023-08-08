// check-pass

// same as #95267, ignore doc comment although it's a bug.

macro_rules! m1 {
    (
        $(
            ///
            $expr: expr,
        )*
    ) => {};
}

m1! {}

macro_rules! m2 {
    (
        $(
            ///
            $expr: expr,
            ///
        )*
    ) => {};
}

m2! {}

macro_rules! m3 {
    (
        $(
            ///
            $tt: tt,
        )*
    ) => {};
}

m3! {}

fn main() {}
