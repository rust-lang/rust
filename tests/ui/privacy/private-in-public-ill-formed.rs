mod aliases_pub {
    struct Priv;
    mod m {
        pub struct Pub3;
    }

    trait PrivTr {
        type AssocAlias;
    }
    impl PrivTr for Priv {
        type AssocAlias = m::Pub3;
    }

    impl <Priv as PrivTr>::AssocAlias {
        pub fn f(arg: Priv) {}
        //~^ ERROR private type `aliases_pub::Priv` in public interface
        //~| ERROR private type `aliases_pub::Priv` in public interface
        //~| ERROR private trait `aliases_pub::PrivTr` in public interface
    }
}

mod aliases_priv {
    struct Priv;
    struct Priv3;

    trait PrivTr {
        type AssocAlias;
    }
    impl PrivTr for Priv {
        type AssocAlias = Priv3;
    }

    impl <Priv as PrivTr>::AssocAlias {
        pub fn f(arg: Priv) {} // OK
    }
}

fn main() {}
