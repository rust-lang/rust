// same as #95267, ignore doc comment although it's a bug.

macro_rules! m1 {
    (
        $(
            ///
        )*
        //~^^^ERROR repetition matches empty token tree
    ) => {};
}

m1! {}

macro_rules! m2 {
    (
        $(
            ///
        )+
        //~^^^ERROR repetition matches empty token tree
    ) => {};
}

m2! {}

macro_rules! m3 {
    (
        $(
            ///
        )?
        //~^^^ERROR repetition matches empty token tree
    ) => {};
}

m3! {}

fn main() {}
