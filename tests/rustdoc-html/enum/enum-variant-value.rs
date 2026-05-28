// This test ensures that the variant value is displayed with underscores but without
// a type name at the end.

//@ aux-build:enum-variant.rs

#![crate_name = "foo"]

extern crate bar;

// In this case, since all variants are C-like variants and at least one of them
// has its value set, we display values for all of them.

//@ has 'foo/enum.A.html'
//@ has - '//*[@class="rust item-decl"]/code' 'A = 12,'
//@ has - '//*[@class="rust item-decl"]/code' 'B = 13,'
//@ has - '//*[@class="rust item-decl"]/code' 'C = 1_245,'
//@ matches - '//*[@id="variant.A"]/h3' '^A = 12$'
//@ matches - '//*[@id="variant.B"]/h3' '^B = 13$'
//@ matches - '//*[@id="variant.C"]/h3' '^C = 1_245$'
pub enum A {
    A = 12,
    B,
    C = 1245,
}

// In this case, all variants are C-like variants but none of them has its value set.
// Therefore we don't display values.

//@ has 'foo/enum.B.html'
//@ has - '//*[@class="rust item-decl"]/code' 'A,'
//@ has - '//*[@class="rust item-decl"]/code' 'B,'
//@ matches - '//*[@id="variant.A"]/h3' '^A$'
//@ matches - '//*[@id="variant.B"]/h3' '^B$'
pub enum B {
    A,
    B,
}

// In this case, not all variants are C-like variants so we don't display values.

//@ has 'foo/enum.C.html'
//@ has - '//*[@class="rust item-decl"]/code' 'A = 12,'
//@ has - '//*[@class="rust item-decl"]/code' 'B,'
//@ has - '//*[@class="rust item-decl"]/code' 'C(u32),'
//@ matches - '//*[@id="variant.A"]/h3' '^A = 12$'
//@ matches - '//*[@id="variant.B"]/h3' '^B$'
//@ has - '//*[@id="variant.C"]/h3' 'C(u32)'
#[repr(u32)]
pub enum C {
    A = 12,
    B,
    C(u32),
}

// In this case, not all variants are C-like variants and no C-like variant has its
// value set, so we don't display values.

//@ has 'foo/enum.D.html'
//@ has - '//*[@class="rust item-decl"]/code' 'A,'
//@ has - '//*[@class="rust item-decl"]/code' 'C(u32),'
//@ matches - '//*[@id="variant.A"]/h3' '^A$'
//@ has - '//*[@id="variant.C"]/h3' 'C(u32)'
pub enum D {
    A,
    C(u32),
}

//@ has 'foo/enum.E.html'
//@ has - '//*[@class="rust item-decl"]/code' 'A = 12,'
//@ has - '//*[@class="rust item-decl"]/code' 'B = 13,'
//@ has - '//*[@class="rust item-decl"]/code' 'C = 1_245,'
//@ matches - '//*[@id="variant.A"]/h3' '^A = 12$'
//@ matches - '//*[@id="variant.B"]/h3' '^B = 13$'
//@ matches - '//*[@id="variant.C"]/h3' '^C = 1_245$'
pub use bar::E;

//@ has 'foo/enum.F.html'
//@ has - '//*[@class="rust item-decl"]/code' 'A,'
//@ has - '//*[@class="rust item-decl"]/code' 'B,'
//@ matches - '//*[@id="variant.A"]/h3' '^A$'
//@ matches - '//*[@id="variant.B"]/h3' '^B$'
pub use bar::F;

//@ has 'foo/enum.G.html'
//@ has - '//*[@class="rust item-decl"]/code' 'A = 12,'
//@ has - '//*[@class="rust item-decl"]/code' 'B,'
//@ has - '//*[@class="rust item-decl"]/code' 'C(u32),'
//@ matches - '//*[@id="variant.A"]/h3' '^A = 12$'
//@ matches - '//*[@id="variant.B"]/h3' '^B$'
//@ has - '//*[@id="variant.C"]/h3' 'C(u32)'
pub use bar::G;

//@ has 'foo/enum.H.html'
//@ has - '//*[@class="rust item-decl"]/code' 'A,'
//@ has - '//*[@class="rust item-decl"]/code' 'C(u32),'
//@ matches - '//*[@id="variant.A"]/h3' '^A$'
//@ has - '//*[@id="variant.C"]/h3' 'C(u32)'
pub use bar::H;

// Testing more complex cases.
pub const X: isize = 2;
//@ has 'foo/enum.I.html'
//@ has - '//*[@class="rust item-decl"]/code' 'A = 2,'
//@ has - '//*[@class="rust item-decl"]/code' 'B = 4,'
//@ has - '//*[@class="rust item-decl"]/code' 'C = 9,'
//@ has - '//*[@class="rust item-decl"]/code' 'D = -1,'
//@ matches - '//*[@id="variant.A"]/h3' '^A = 2$'
//@ matches - '//*[@id="variant.B"]/h3' '^B = 4$'
//@ matches - '//*[@id="variant.C"]/h3' '^C = 9$'
//@ matches - '//*[@id="variant.D"]/h3' '^D = -1$'
#[repr(isize)]
pub enum I {
    A = X,
    B = X * 2,
    C = Self::B as isize + X + 3,
    D = -1,
}

// Testing `repr`.

//@ has 'foo/enum.J.html'
//@ has - '//*[@class="rust item-decl"]/code' 'A = 0,'
//@ has - '//*[@class="rust item-decl"]/code' 'B = 1,'
//@ matches - '//*[@id="variant.A"]/h3' '^A = 0$'
//@ matches - '//*[@id="variant.B"]/h3' '^B = 1$'
#[repr(C)]
pub enum J {
    A,
    B,
}

//@ has 'foo/enum.K.html'
//@ has - '//*[@class="rust item-decl"]/code' 'A(u32),'
//@ has - '//*[@class="rust item-decl"]/code' 'B,'
//@ has - '//*[@id="variant.A"]/h3' 'A(u32)'
//@ matches - '//*[@id="variant.B"]/h3' '^B$'
#[repr(C)]
pub enum K {
    A(u32),
    B,
}

//@ has 'foo/enum.L.html'
//@ has - '//*[@class="rust item-decl"]/code' 'A = 0,'
//@ has - '//*[@class="rust item-decl"]/code' 'B = 1,'
//@ matches - '//*[@id="variant.A"]/h3' '^A = 0$'
//@ matches - '//*[@id="variant.B"]/h3' '^B = 1$'
#[repr(u32)]
pub enum L {
    A,
    B,
}

//@ has 'foo/enum.M.html'
//@ has - '//*[@class="rust item-decl"]/code' 'A(u32),'
//@ has - '//*[@class="rust item-decl"]/code' 'B,'
//@ has - '//*[@id="variant.A"]/h3' 'A(u32)'
//@ matches - '//*[@id="variant.B"]/h3' '^B$'
#[repr(u32)]
pub enum M {
    A(u32),
    B,
}

//@ has 'foo/enum.N.html'
//@ has - '//*[@class="rust item-decl"]/code' 'A = 0,'
//@ has - '//*[@class="rust item-decl"]/code' 'B = 1,'
//@ matches - '//*[@id="variant.A"]/h3' '^A = 0$'
//@ matches - '//*[@id="variant.B"]/h3' '^B = 1$'
pub use bar::N;

//@ has 'foo/enum.O.html'
//@ has - '//*[@class="rust item-decl"]/code' 'A(u32),'
//@ has - '//*[@class="rust item-decl"]/code' 'B,'
//@ has - '//*[@id="variant.A"]/h3' 'A(u32)'
//@ matches - '//*[@id="variant.B"]/h3' '^B$'
pub use bar::O;

//@ has 'foo/enum.P.html'
//@ has - '//*[@class="rust item-decl"]/code' 'A = 0,'
//@ has - '//*[@class="rust item-decl"]/code' 'B = 1,'
//@ matches - '//*[@id="variant.A"]/h3' '^A = 0$'
//@ matches - '//*[@id="variant.B"]/h3' '^B = 1$'
pub use bar::P;

//@ has 'foo/enum.Q.html'
//@ has - '//*[@class="rust item-decl"]/code' 'A(u32),'
//@ has - '//*[@class="rust item-decl"]/code' 'B,'
//@ has - '//*[@id="variant.A"]/h3' 'A(u32)'
//@ matches - '//*[@id="variant.B"]/h3' '^B$'
pub use bar::Q;

// Ensure signed implicit discriminants are rendered correctly after a negative explicit value.
//@ has 'foo/enum.R.html'
//@ has - '//*[@class="rust item-decl"]/code' 'A = -2,'
//@ has - '//*[@class="rust item-decl"]/code' 'B = -1,'
//@ matches - '//*[@id="variant.A"]/h3' '^A = -2$'
//@ matches - '//*[@id="variant.B"]/h3' '^B = -1$'
pub enum R {
    A = -2,
    B,
}

// Also check that incrementing -1 yields 0 for the next implicit variant.
//@ has 'foo/enum.S.html'
//@ has - '//*[@class="rust item-decl"]/code' 'A = -1,'
//@ has - '//*[@class="rust item-decl"]/code' 'B = 0,'
//@ matches - '//*[@id="variant.A"]/h3' '^A = -1$'
//@ matches - '//*[@id="variant.B"]/h3' '^B = 0$'
pub enum S {
    A = -1,
    B,
}
