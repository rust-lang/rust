// Make sure we produce the unfulfilled expectation lint exactly once if neither the
// struct nor the derived code fulfilled it: one written attribute is one expectation,
// no matter how many impls are derived from the item.

//@ check-pass

#[expect(unexpected_cfgs)]
//~^ WARN this lint expectation is unfulfilled
#[derive(Debug)]
pub struct MyStruct {
    pub t_ref: i64,
}

fn main() {}
