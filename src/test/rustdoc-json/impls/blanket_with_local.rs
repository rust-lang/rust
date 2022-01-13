// Test for the ICE in rust/83718
// A blanket impl plus a local type together shouldn't result in mismatched ID issues

// @has method_abi.json "$.index[*][?(@.name=='Load')]"
pub trait Load {
    fn load() {}
}

impl<P> Load for P {
    fn load() {}
}

// @has - "$.index[*][?(@.name=='Wrapper')]"
pub struct Wrapper {}
