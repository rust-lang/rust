// FIXME: Bring back duplication of the `#[expect]` attribute when deriving.
//
// Make sure we produce the unfulfilled expectation lint if neither the struct or the
// derived code fulfilled it.

//@ check-pass

#[expect(unexpected_cfgs)]
//~^ WARN this lint expectation is unfulfilled
//FIXME ~^^ WARN this lint expectation is unfulfilled
#[derive(Debug)]
pub struct MyStruct {
    pub t_ref: i64,
}

fn main() {}
