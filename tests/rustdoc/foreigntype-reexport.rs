#![feature(extern_types)]

mod sub {
    extern "C" {
        /// Another extern type.
        pub type C2;
        pub fn f2();
        pub static K: usize;
    }
}

pub mod sub2 {
    extern "C" {
        // @has foreigntype_reexport/sub2/foreigntype.C.html
        pub type C;
        // @has foreigntype_reexport/sub2/fn.f.html
        pub fn f();
        // @has foreigntype_reexport/sub2/static.K3.html
        pub static K3: usize;
    }
}

mod sub3 {
    extern "C" {
        pub type C4;
        pub fn f4();
        pub static K4: usize;
        type X4;
    }
}

// @has foreigntype_reexport/foreigntype.C2.html
// @has foreigntype_reexport/fn.f2.html
// @has foreigntype_reexport/static.K2.html
// @has foreigntype_reexport/index.html '//a[@class="foreigntype"]' 'C2'
// @has foreigntype_reexport/index.html '//a[@class="fn"]' 'f2'
// @has foreigntype_reexport/index.html '//a[@class="static"]' 'K2'
pub use self::sub::{f2, C2, K as K2};

// @has foreigntype_reexport/index.html '//a[@class="foreigntype"]' 'C'
// @has foreigntype_reexport/index.html '//a[@class="fn"]' 'f'
// @has foreigntype_reexport/index.html '//a[@class="static"]' 'K3'
// @has foreigntype_reexport/index.html '//code' 'pub use self::sub2::C as C3;'
// @has foreigntype_reexport/index.html '//code' 'pub use self::sub2::f as f3;'
// @has foreigntype_reexport/index.html '//code' 'pub use self::sub2::K3;'
pub use self::sub2::{f as f3, C as C3, K3};

// @has foreigntype_reexport/foreigntype.C4.html
// @has foreigntype_reexport/fn.f4.html
// @has foreigntype_reexport/static.K4.html
// @!has foreigntype_reexport/foreigntype.X4.html
// @has foreigntype_reexport/index.html '//a[@class="foreigntype"]' 'C4'
// @has foreigntype_reexport/index.html '//a[@class="fn"]' 'f4'
// @has foreigntype_reexport/index.html '//a[@class="static"]' 'K4'
// @!has foreigntype_reexport/index.html '//a[@class="foreigntype"]' 'X4'
pub use self::sub3::*;
