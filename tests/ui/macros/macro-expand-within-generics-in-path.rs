// issue#123911
// issue#123912

macro_rules! m {
    ($p: path) => {
        #[$p]
        struct S;
    };
}

macro_rules! p {
    () => {};
}

m!(generic<p!()>);
//~^ ERROR: unexpected generic arguments in path
//~| ERROR: cannot find attribute `generic` in this scope

fn main() {}
