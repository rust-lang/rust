mod fn_scope;
mod mod_scope;

pub use self::{
    fn_scope::{resolve_local_name, FnScopes},
    mod_scope::ModuleScope,
};
