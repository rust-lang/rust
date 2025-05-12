mod options {
    pub struct ParseOptions {}
}

mod parser {
    pub use options::*;
    // Private single import shadows public glob import, but arrives too late for initial
    // resolution of `use parser::ParseOptions` because it depends on that resolution itself.
    #[allow(hidden_glob_reexports)]
    use ParseOptions;
}

pub use parser::ParseOptions; //~ ERROR struct import `ParseOptions` is private

fn main() {}
