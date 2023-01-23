#![crate_name = "foo"]

// @has 'foo/type.Resolutions.html'
// @has - '//div[@class="item-decl"]/pre[@class="rust"]' "pub type Resolutions<'tcx> = &'tcx u8;"
pub type Resolutions<'tcx> = &'tcx u8;
