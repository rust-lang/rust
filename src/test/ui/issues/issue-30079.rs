struct SemiPriv;

mod m1 {
    struct Priv;
    impl ::SemiPriv {
        pub fn f(_: Priv) {} //~ WARN private type `m1::Priv` in public interface
        //~^ WARNING hard error
    }

    impl Priv {
        pub fn f(_: Priv) {} // ok
    }
}

mod m2 {
    struct Priv;
    impl ::std::ops::Deref for ::SemiPriv {
        type Target = Priv; //~ ERROR private type `m2::Priv` in public interface
        fn deref(&self) -> &Self::Target { unimplemented!() }
    }

    impl ::std::ops::Deref for Priv {
        type Target = Priv; // ok
        fn deref(&self) -> &Self::Target { unimplemented!() }
    }
}

trait SemiPrivTrait {
    type Assoc;
}

mod m3 {
    struct Priv;
    impl ::SemiPrivTrait for () {
        type Assoc = Priv; //~ ERROR private type `m3::Priv` in public interface
    }
}

fn main() {}
