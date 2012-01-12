// error-pattern:empty #[link_name] not allowed; use #[nolink].
// Issue #1326

#[link_name = ""]
#[nolink]
native mod foo {
}
