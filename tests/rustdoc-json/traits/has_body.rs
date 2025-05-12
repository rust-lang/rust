//@ has "$.index[?(@.name=='Foo')]"
pub trait Foo {
    //@ is "$.index[?(@.name=='no_self')].inner.function.has_body" false
    fn no_self();
    //@ is "$.index[?(@.name=='move_self')].inner.function.has_body" false
    fn move_self(self);
    //@ is "$.index[?(@.name=='ref_self')].inner.function.has_body" false
    fn ref_self(&self);

    //@ is "$.index[?(@.name=='no_self_def')].inner.function.has_body" true
    fn no_self_def() {}
    //@ is "$.index[?(@.name=='move_self_def')].inner.function.has_body" true
    fn move_self_def(self) {}
    //@ is "$.index[?(@.name=='ref_self_def')].inner.function.has_body" true
    fn ref_self_def(&self) {}
}

pub trait Bar: Clone {
    //@ is "$.index[?(@.name=='method')].inner.function.has_body" false
    fn method(&self, param: usize);
}
