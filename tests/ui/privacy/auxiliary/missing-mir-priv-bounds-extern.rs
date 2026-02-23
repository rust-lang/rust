pub trait ToPriv {
    type AssocPriv;
}

pub trait PubTr {
    #[expect(private_bounds)]
    type Assoc: ToPriv<AssocPriv = Priv>;
}

struct Dummy;
struct DummyToPriv;
impl PubTr for Dummy {
    type Assoc = DummyToPriv;
}
impl ToPriv for DummyToPriv {
    type AssocPriv = Priv;
}

pub fn get_dummy() -> impl PubTr {
    Dummy
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
