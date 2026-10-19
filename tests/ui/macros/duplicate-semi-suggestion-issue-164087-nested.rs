macro_rules! inner {
    () => {;}; //~ ERROR: expected expression, found `;`
}

macro_rules! outer {
    () => {{ let _ = inner!(); }};
}

fn main() {
    outer!()
}
