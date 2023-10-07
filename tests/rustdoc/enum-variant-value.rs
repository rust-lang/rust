// This test ensures that the variant value is displayed with underscores but without
// a type name at the end.

// aux-build:enum-variant.rs

#![crate_name = "foo"]

extern crate bar;

// In this case, since all variants are C-like variants and at least one of them
// has its value set, we display values for all of them.

// @has 'foo/enum.A.html'
// @has - '//*[@class="rust item-decl"]/code' 'A = 12,'
// @has - '//*[@class="rust item-decl"]/code' 'B = 13,'
// @has - '//*[@class="rust item-decl"]/code' 'C = 1_245,'
// @matches - '//*[@id="variant.A"]/h3' '^A = 12$'
// @matches - '//*[@id="variant.B"]/h3' '^B = 13$'
// @matches - '//*[@id="variant.C"]/h3' '^C = 1_245$'
pub enum A {
    A = 12,
    B,
    C = 1245,
}

// In this case, all variants are C-like variants but none of them has its value set.
// Therefore we don't display values.

// @has 'foo/enum.B.html'
// @has - '//*[@class="rust item-decl"]/code' 'A,'
// @has - '//*[@class="rust item-decl"]/code' 'B,'
// @matches - '//*[@id="variant.A"]/h3' '^A$'
// @matches - '//*[@id="variant.B"]/h3' '^B$'
pub enum B {
    A,
    B,
}

// In this case, not all variants are C-like variants so we don't display values.

// @has 'foo/enum.C.html'
// @has - '//*[@class="rust item-decl"]/code' 'A = 12,'
// @has - '//*[@class="rust item-decl"]/code' 'B,'
// @has - '//*[@class="rust item-decl"]/code' 'C(u32),'
// @matches - '//*[@id="variant.A"]/h3' '^A = 12$'
// @matches - '//*[@id="variant.B"]/h3' '^B$'
// @has - '//*[@id="variant.C"]/h3' 'C(u32)'
#[repr(u32)]
pub enum C {
    A = 12,
    B,
    C(u32),
}

// In this case, not all variants are C-like variants and no C-like variant has its
// value set, so we don't display values.

// @has 'foo/enum.D.html'
// @has - '//*[@class="rust item-decl"]/code' 'A,'
// @has - '//*[@class="rust item-decl"]/code' 'C(u32),'
// @matches - '//*[@id="variant.A"]/h3' '^A$'
// @has - '//*[@id="variant.C"]/h3' 'C(u32)'
pub enum D {
    A,
    C(u32),
}

// @has 'foo/enum.E.html'
// @has - '//*[@class="rust item-decl"]/code' 'A = 12,'
// @has - '//*[@class="rust item-decl"]/code' 'B = 13,'
// @has - '//*[@class="rust item-decl"]/code' 'C = 1_245,'
// @matches - '//*[@id="variant.A"]/h3' '^A = 12$'
// @matches - '//*[@id="variant.B"]/h3' '^B = 13$'
// @matches - '//*[@id="variant.C"]/h3' '^C = 1_245$'
pub use bar::E;

// @has 'foo/enum.F.html'
// @has - '//*[@class="rust item-decl"]/code' 'A,'
// @has - '//*[@class="rust item-decl"]/code' 'B,'
// @matches - '//*[@id="variant.A"]/h3' '^A$'
// @matches - '//*[@id="variant.B"]/h3' '^B$'
pub use bar::F;

// @has 'foo/enum.G.html'
// @has - '//*[@class="rust item-decl"]/code' 'A = 12,'
// @has - '//*[@class="rust item-decl"]/code' 'B,'
// @has - '//*[@class="rust item-decl"]/code' 'C(u32),'
// @matches - '//*[@id="variant.A"]/h3' '^A = 12$'
// @matches - '//*[@id="variant.B"]/h3' '^B$'
// @has - '//*[@id="variant.C"]/h3' 'C(u32)'
pub use bar::G;

// @has 'foo/enum.H.html'
// @has - '//*[@class="rust item-decl"]/code' 'A,'
// @has - '//*[@class="rust item-decl"]/code' 'C(u32),'
// @matches - '//*[@id="variant.A"]/h3' '^A$'
// @has - '//*[@id="variant.C"]/h3' 'C(u32)'
pub use bar::H;
