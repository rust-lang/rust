#![crate_name = "foo"]

// @has foo/type.CellIndex.html '//div[@class="item-decl"]/pre[@class="rust"]' 'type CellIndex<const D: usize> = [i64; D];'
pub type CellIndex<const D: usize> = [i64; D];
