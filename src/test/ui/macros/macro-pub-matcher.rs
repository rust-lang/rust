// run-pass
#![allow(dead_code, unused_imports)]
#![feature(crate_visibility_modifier)]

/**
Ensure that `:vis` matches can be captured in existing positions, and passed
through without the need for reparse tricks.
*/
macro_rules! vis_passthru {
    ($vis:vis const $name:ident: $ty:ty = $e:expr;) => { $vis const $name: $ty = $e; };
    ($vis:vis enum $name:ident {}) => { $vis struct $name {} };
    ($vis:vis extern "C" fn $name:ident() {}) => { $vis extern "C" fn $name() {} };
    ($vis:vis fn $name:ident() {}) => { $vis fn $name() {} };
    ($vis:vis mod $name:ident {}) => { $vis mod $name {} };
    ($vis:vis static $name:ident: $ty:ty = $e:expr;) => { $vis static $name: $ty = $e; };
    ($vis:vis struct $name:ident;) => { $vis struct $name; };
    ($vis:vis trait $name:ident {}) => { $vis trait $name {} };
    ($vis:vis type $name:ident = $ty:ty;) => { $vis type $name = $ty; };
    ($vis:vis use $path:ident as $name:ident;) => { $vis use self::$path as $name; };
}

mod with_pub {
    vis_passthru! { pub const A: i32 = 0; }
    vis_passthru! { pub enum B {} }
    vis_passthru! { pub extern "C" fn c() {} }
    vis_passthru! { pub mod d {} }
    vis_passthru! { pub static E: i32 = 0; }
    vis_passthru! { pub struct F; }
    vis_passthru! { pub trait G {} }
    vis_passthru! { pub type H = i32; }
    vis_passthru! { pub use A as I; }
}

mod without_pub {
    vis_passthru! { const A: i32 = 0; }
    vis_passthru! { enum B {} }
    vis_passthru! { extern "C" fn c() {} }
    vis_passthru! { mod d {} }
    vis_passthru! { static E: i32 = 0; }
    vis_passthru! { struct F; }
    vis_passthru! { trait G {} }
    vis_passthru! { type H = i32; }
    vis_passthru! { use A as I; }
}

mod with_pub_restricted {
    vis_passthru! { pub(crate) const A: i32 = 0; }
    vis_passthru! { pub(crate) enum B {} }
    vis_passthru! { pub(crate) extern "C" fn c() {} }
    vis_passthru! { pub(crate) mod d {} }
    vis_passthru! { pub(crate) static E: i32 = 0; }
    vis_passthru! { pub(crate) struct F; }
    vis_passthru! { pub(crate) trait G {} }
    vis_passthru! { pub(crate) type H = i32; }
    vis_passthru! { pub(crate) use A as I; }
}

mod with_crate {
    vis_passthru! { crate const A: i32 = 0; }
    vis_passthru! { crate enum B {} }
    vis_passthru! { crate extern "C" fn c() {} }
    vis_passthru! { crate mod d {} }
    vis_passthru! { crate static E: i32 = 0; }
    vis_passthru! { crate struct F; }
    vis_passthru! { crate trait G {} }
    vis_passthru! { crate type H = i32; }
    vis_passthru! { crate use A as I; }
}

mod garden {
    mod with_pub_restricted_path {
        vis_passthru! { pub(in garden) const A: i32 = 0; }
        vis_passthru! { pub(in garden) enum B {} }
        vis_passthru! { pub(in garden) extern "C" fn c() {} }
        vis_passthru! { pub(in garden) mod d {} }
        vis_passthru! { pub(in garden) static E: i32 = 0; }
        vis_passthru! { pub(in garden) struct F; }
        vis_passthru! { pub(in garden) trait G {} }
        vis_passthru! { pub(in garden) type H = i32; }
        vis_passthru! { pub(in garden) use A as I; }
    }
}

/*
Ensure that the `:vis` matcher works in a more complex situation: parsing a
struct definition.
*/
macro_rules! vis_parse_struct {
    ($(#[$($attrs:tt)*])* $vis:vis struct $name:ident {$($body:tt)*}) => {
        vis_parse_struct! { @parse_fields $(#[$($attrs)*])*, $vis, $name, $($body)* }
    };

    ($(#[$($attrs:tt)*])* $vis:vis struct $name:ident ($($body:tt)*);) => {
        vis_parse_struct! { @parse_tuple $(#[$($attrs)*])*, $vis, $name, $($body)* }
    };

    (@parse_fields
     $(#[$attrs:meta])*, $vis:vis, $name:ident, $($fvis:vis $fname:ident: $fty:ty),* $(,)*) => {
        $(#[$attrs])* $vis struct $name { $($fvis $fname: $fty,)* }
    };

    (@parse_tuple
     $(#[$attrs:meta])*, $vis:vis, $name:ident, $($fvis:vis $fty:ty),* $(,)*) => {
        $(#[$attrs])* $vis struct $name ( $($fvis $fty,)* );
    };
}

mod test_struct {
    vis_parse_struct! { pub(crate) struct A { pub a: i32, b: i32, pub(crate) c: i32 } }
    vis_parse_struct! { pub struct B { a: i32, pub(crate) b: i32, pub c: i32 } }
    vis_parse_struct! { struct C { pub(crate) a: i32, pub b: i32, c: i32 } }

    vis_parse_struct! { pub(crate) struct D (pub i32, i32, pub(crate) i32); }
    vis_parse_struct! { pub struct E (i32, pub(crate) i32, pub i32); }
    vis_parse_struct! { struct F (pub(crate) i32, pub i32, i32); }
}

fn main() {}
