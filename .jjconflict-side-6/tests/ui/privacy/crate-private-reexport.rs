fn f1() {}
enum E1 { V }
struct S1 {
    #[rustfmt::skip]
    bar: i32,
}
mod m1 {
    pub use crate::f1; //~ ERROR `f1` is only public within the crate, and cannot be re-exported outside
    pub use crate::S1; //~ ERROR `S1` is only public within the crate, and cannot be re-exported outside
    pub use crate::E1; //~ ERROR `E1` is only public within the crate, and cannot be re-exported outside
    pub use crate::E1::V; //~ ERROR `V` is only public within the crate, and cannot be re-exported outside
}

pub(crate) fn f2() {}
pub(crate) enum E2 {
    V
}
pub(crate) struct S2 {
    #[rustfmt::skip]
    bar: i32,
}
mod m2 {
    pub use crate::f2; //~ ERROR `f2` is only public within the crate, and cannot be re-exported outside
    pub use crate::S2; //~ ERROR `S2` is only public within the crate, and cannot be re-exported outside
    pub use crate::E2; //~ ERROR `E2` is only public within the crate, and cannot be re-exported outside
    pub use crate::E2::V; //~ ERROR `V` is only public within the crate, and cannot be re-exported outside
}

mod m3 {
    pub(crate) fn f3() {}
    pub(crate) enum E3 {
        V
    }
    pub(crate) struct S3 {
        #[rustfmt::skip]
        bar: i32,
    }
}
pub use m3::f3; //~ ERROR `f3` is only public within the crate, and cannot be re-exported outside
pub use m3::S3; //~ ERROR `S3` is only public within the crate, and cannot be re-exported outside
pub use m3::E3; //~ ERROR `E3` is only public within the crate, and cannot be re-exported outside
pub use m3::E3::V; //~ ERROR `V` is only public within the crate, and cannot be re-exported outside

pub(self) fn f4() {}
pub use crate::f4 as f5; //~ ERROR `f4` is only public within the crate, and cannot be re-exported outside

pub mod m10 {
    pub mod m {
        pub(super) fn f6() {}
        pub(crate) fn f7() {}
        pub(in crate::m10) fn f8() {}
    }
    pub use self::m::f6; //~ ERROR `f6` is private, and cannot be re-exported
    pub use self::m::f7; //~ ERROR `f7` is only public within the crate, and cannot be re-exported outside
    pub use self::m::f8; //~ ERROR `f8` is private, and cannot be re-exported
}
pub use m10::m::f6; //~ ERROR function `f6` is private
pub use m10::m::f7; //~ ERROR `f7` is only public within the crate, and cannot be re-exported outside
pub use m10::m::f8; //~ ERROR function `f8` is private

pub mod m11 {
    pub(self) fn f9() {}
}
pub use m11::f9; //~ ERROR function `f9` is private

fn main() {}
