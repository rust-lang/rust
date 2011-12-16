// error-pattern:empty #[link_name] not allowed; use #[nolink].

#[link_name = ""]
native mod foo {
}
