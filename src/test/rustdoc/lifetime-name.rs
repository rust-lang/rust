#![crate_name = "foo"]

// @has 'foo/type.Resolutions.html'
// @has - '//*[@class="rust typedef"]' "pub type Resolutions<'tcx> = &'tcx u8;"
pub type Resolutions<'tcx> = &'tcx u8;
