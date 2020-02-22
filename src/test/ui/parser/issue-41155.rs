struct S;

impl S {
    pub //~ ERROR unmatched visibility `pub`
} //~ ERROR non-item in item list

fn main() {}
