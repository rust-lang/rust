struct S;

impl S {
    pub //~ ERROR visibility `pub` is not followed by an item
} //~ ERROR non-item in item list

fn main() {}
