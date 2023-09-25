// This test ensures that the variant value is displayed with underscores but without
// a type name at the end.

#![crate_name = "foo"]

// @has 'foo/enum.B.html'
// @has - '//*[@class="rust item-decl"]/code' 'A = 12,'
// @has - '//*[@class="rust item-decl"]/code' 'C = 1_245,'
// @matches - '//*[@id="variant.A"]/h3' '^A = 12$'
// @matches - '//*[@id="variant.C"]/h3' '^C = 1_245$'
pub enum B {
    A = 12,
    B,
    C = 1245,
}
