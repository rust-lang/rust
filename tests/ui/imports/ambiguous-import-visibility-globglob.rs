//@ check-pass

// FIXME: should report "ambiguous import visibility" in all the cases below.

mod m {
    pub struct S {}
}

mod min_vis_first {
    use crate::m::*;
    pub(crate) use crate::m::*;
    pub use crate::m::*;

    pub use self::S as S1;
    pub(crate) use self::S as S2;
    use self::S as S3; // OK
}

mod mid_vis_first {
    pub(crate) use crate::m::*;
    use crate::m::*;
    pub use crate::m::*;

    pub use self::S as S1;
    pub(crate) use self::S as S2;
    use self::S as S3; // OK
}

mod max_vis_first {
    pub use crate::m::*;
    use crate::m::*;
    pub(crate) use crate::m::*;

    pub use self::S as S1;
    pub(crate) use self::S as S2;
    use self::S as S3; // OK
}

fn main() {}
