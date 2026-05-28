#![crate_name = "foo"]

//@ has foo/type.CellIndex.html '//pre[@class="rust item-decl"]' 'type CellIndex<const D: usize> = [i64; D];'
pub type CellIndex<const D: usize> = [i64; D];
