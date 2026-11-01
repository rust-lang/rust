// Doesn't involve `impl Trait` unlike missing-mir-priv-bounds-extern.rs

pub trait ToPriv {
    type AssocPriv;
}

pub trait PubTr {
    #[expect(private_bounds)]
    type Assoc: ToPriv<AssocPriv = Priv>;
}

// Dummy and DummyToPriv are only used in call_handler
struct Dummy;
struct DummyToPriv;
impl PubTr for Dummy {
    type Assoc = DummyToPriv;
}
impl ToPriv for DummyToPriv {
    type AssocPriv = Priv;
}

pub trait PubTrHandler {
    fn handle<T: PubTr>();
}
pub fn call_handler<T: PubTrHandler>() {
    T::handle::<Dummy>();
}

struct Priv;

pub trait GetUnreachable {
    type Assoc;
}

mod m {
    pub struct Unreachable;

    impl Unreachable {
        #[expect(dead_code)]
        pub fn generic<T>() {}
    }

    impl crate::GetUnreachable for crate::Priv {
        type Assoc = Unreachable;
    }
}
