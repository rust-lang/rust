mod m {
    struct Priv;

    #[allow(private_interfaces)]
    pub struct PubStruct(pub Priv);
    pub enum PubEnum {
        #[allow(private_interfaces)]
        Variant(Priv),
    }

    impl PubStruct {
        pub fn new() -> PubStruct {
            PubStruct(Priv)
        }
    }
    impl PubEnum {
        pub fn new() -> PubEnum {
            PubEnum::Variant(Priv)
        }
    }
}

fn main() {
    match m::PubStruct::new() {
        m::PubStruct(_) => {} //~ ERROR type `Priv` is private
        m::PubStruct(..) => {} //~ ERROR type `Priv` is private
    }

    match m::PubEnum::new() {
        m::PubEnum::Variant(_) => {} //~ ERROR type `Priv` is private
        m::PubEnum::Variant(..) => {} //~ ERROR type `Priv` is private
    }

    let _ = m::PubStruct; //~ ERROR type `Priv` is private
    let _ = m::PubEnum::Variant; //~ ERROR type `Priv` is private
}
