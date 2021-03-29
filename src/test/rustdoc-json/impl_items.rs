
#![no_std]

// @has impl_items.json "$.index[*][?(@.name=='Simple')]"
pub struct Simple;

impl Simple {
    // @is - "$.index[*][?(@.name=='CONSTANT')].kind" \"constant\"
    pub const CONSTANT: usize = 0;
}

// @has - "$.index[*][?(@.name=='EasyImpl')]"
pub trait EasyImpl {
    // @has - "$.index[*][?(@.name=='DeclareMe')].kind" \"assoc_type\"
    type DeclareMe;
    // @has - "$.index[*][?(@.name=='ASSIGN_ME')].kind" \"assoc_const\"
    const ASSIGN_ME: usize;
}

impl EasyImpl for Simple {
    // @has - "$.index[*][?(@.name=='DeclareMe')].kind" \"typedef\"
    type DeclareMe = usize;
    // @has - "$.index[*][?(@.name=='ASSIGN_ME')].kind" \"constant\"
    const ASSIGN_ME: usize = 0;
}
