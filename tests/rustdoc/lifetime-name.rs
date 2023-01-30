#![crate_name = "foo"]

// @has 'foo/type.Resolutions.html'
// @has - '//pre[@class="rust item-decl"]' "pub type Resolutions<'tcx> = &'tcx u8;"
pub type Resolutions<'tcx> = &'tcx u8;
