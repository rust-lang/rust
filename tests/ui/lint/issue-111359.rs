#[deny(missing_debug_implementations)]
#[deny(missing_copy_implementations)]

mod priv_mod {
    use std::convert::TryFrom;

    pub struct BarPub;
    //~^ ERROR type does not implement `Debug`; consider adding `#[derive(Debug)]` or a manual implementation
    //~| ERROR type could implement `Copy`; consider adding `impl Copy`
    struct BarPriv;

    impl<'a> TryFrom<BarPriv> for u8 {
        type Error = ();
        fn try_from(o: BarPriv) -> Result<Self, ()> {
            unimplemented!()
        }
    }

    impl<'a> TryFrom<BarPub> for u8 {
        type Error = ();
        fn try_from(o: BarPub) -> Result<Self, ()> {
            unimplemented!()
        }
    }
}

fn main() {}
