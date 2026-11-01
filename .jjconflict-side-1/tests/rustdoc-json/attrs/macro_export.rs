//@ compile-flags: --document-private-items

//@ set exported_id = "$.index[?(@.name=='exported')].id"
//@ is "$.index[?(@.name=='exported')].attrs" '["macro_export"]'
//@ is "$.index[?(@.name=='exported')].visibility" '"public"'

#[macro_export]
macro_rules! exported {
    () => {};
}

//@ set not_exported_id = "$.index[?(@.name=='not_exported')].id"
//@ is "$.index[?(@.name=='not_exported')].attrs" []
//@ is "$.index[?(@.name=='not_exported')].visibility" '"crate"'
macro_rules! not_exported {
    () => {};
}

//@ set module_id = "$.index[?(@.name=='module')].id"
pub mod module {
    //@ set exported_from_mod_id = "$.index[?(@.name=='exported_from_mod')].id"
    //@ is "$.index[?(@.name=='exported_from_mod')].attrs" '["macro_export"]'
    //@ is "$.index[?(@.name=='exported_from_mod')].visibility" '"public"'
    #[macro_export]
    macro_rules! exported_from_mod {
        () => {};
    }

    //@ set not_exported_from_mod_id = "$.index[?(@.name=='not_exported_from_mod')].id"
    //@ is "$.index[?(@.name=='not_exported_from_mod')].attrs" []
    //@ is "$.index[?(@.name=='not_exported_from_mod')].visibility" '"crate"'
    macro_rules! not_exported_from_mod {
        () => {};
    }
}
// The non-exported macro's are left in place, but the #[macro_export]'d ones
// are moved to the crate root.

//@ is "$.index[?(@.name=='module')].inner.module.items[*]" $not_exported_from_mod_id
//@ ismany "$.index[?(@.name=='macro_export')].inner.module.items[*]" $exported_id $not_exported_id $module_id $exported_from_mod_id
