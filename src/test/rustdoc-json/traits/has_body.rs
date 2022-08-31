// @has "$.index[*][?(@.name=='Foo')]"
pub trait Foo {
    // @has "$.index[*][?(@.name=='no_self')].inner.has_body" false
    fn no_self();
    // @has "$.index[*][?(@.name=='move_self')].inner.has_body" false
    fn move_self(self);
    // @has "$.index[*][?(@.name=='ref_self')].inner.has_body" false
    fn ref_self(&self);

    // @has "$.index[*][?(@.name=='no_self_def')].inner.has_body" true
    fn no_self_def() {}
    // @has "$.index[*][?(@.name=='move_self_def')].inner.has_body" true
    fn move_self_def(self) {}
    // @has "$.index[*][?(@.name=='ref_self_def')].inner.has_body" true
    fn ref_self_def(&self) {}
}

pub trait Bar: Clone {
    // @has "$.index[*][?(@.name=='method')].inner.has_body" false
    fn method(&self, param: usize);
}
