mod m {
    pub struct S {}
}

mod min_vis_first {
    use crate::m::*;
    pub(crate) use crate::m::*;
    pub use crate::m::*;

    pub use self::S as S1;
    //~^ ERROR ambiguous import visibility
    //~| WARN this was previously accepted
    pub(crate) use self::S as S2;
    //~^ ERROR ambiguous import visibility
    //~| WARN this was previously accepted
    use self::S as S3; // OK
}

mod mid_vis_first {
    pub(crate) use crate::m::*;
    use crate::m::*;
    pub use crate::m::*;

    pub use self::S as S1;
    //~^ ERROR ambiguous import visibility
    //~| WARN this was previously accepted
    pub(crate) use self::S as S2;
    //~^ ERROR ambiguous import visibility
    //~| WARN this was previously accepted
    use self::S as S3; // OK
}

mod max_vis_first {
    pub use crate::m::*;
    use crate::m::*;
    pub(crate) use crate::m::*;

    pub use self::S as S1;
    //~^ ERROR ambiguous import visibility
    //~| WARN this was previously accepted
    pub(crate) use self::S as S2;
    //~^ ERROR ambiguous import visibility
    //~| WARN this was previously accepted
    use self::S as S3; // OK
}

fn main() {}
