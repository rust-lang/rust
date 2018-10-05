mod fn_scope;
mod mod_scope;

pub use self::{
    fn_scope::{FnScopes, resolve_local_name},
    mod_scope::ModuleScope,
};

