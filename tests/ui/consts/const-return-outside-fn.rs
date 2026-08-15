//! regression test for issue <https://github.com/rust-lang/rust/issues/38458>
const x: () = {
    return; //~ ERROR return statement outside of function body
};

fn main() {}
