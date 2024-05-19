fn main() {}

macro_rules! ambiguity {
    ($($i:ident)* $j:ident) => {};
}

ambiguity!(error); //~ ERROR local ambiguity
ambiguity!(error); //~ ERROR local ambiguity
