// error-pattern:empty #[link_name] not allowed; use #[nolink].

#[link_name = ""]
extern mod foo {
    #[legacy_exports];
}
