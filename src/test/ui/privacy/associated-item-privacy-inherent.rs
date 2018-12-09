#![feature(decl_macro, associated_type_defaults)]
#![allow(unused, private_in_public)]

mod priv_nominal {
    pub struct Pub;
    impl Pub {
        fn method(&self) {}
        const CONST: u8 = 0;
        // type AssocTy = u8;
    }

    pub macro mac() {
        let value = Pub::method;
        //~^ ERROR type `for<'r> fn(&'r priv_nominal::Pub) {<priv_nominal::Pub>::method}` is private
        value;
        //~^ ERROR type `for<'r> fn(&'r priv_nominal::Pub) {<priv_nominal::Pub>::method}` is private
        Pub.method();
        //~^ ERROR type `for<'r> fn(&'r priv_nominal::Pub) {<priv_nominal::Pub>::method}` is private
        Pub::CONST;
        //~^ ERROR associated constant `CONST` is private
        // let _: Pub::AssocTy;
        // pub type InSignatureTy = Pub::AssocTy;
    }
}
fn priv_nominal() {
    priv_nominal::mac!();
}

mod priv_signature {
    struct Priv;
    pub struct Pub;
    impl Pub {
        pub fn method(&self, arg: Priv) {}
    }

    pub macro mac() {
        let value = Pub::method;
        //~^ ERROR type `priv_signature::Priv` is private
        value;
        //~^ ERROR type `priv_signature::Priv` is private
        Pub.method(loop {});
        //~^ ERROR type `priv_signature::Priv` is private
    }
}
fn priv_signature() {
    priv_signature::mac!();
}

mod priv_substs {
    struct Priv;
    pub struct Pub;
    impl Pub {
        pub fn method<T>(&self) {}
    }

    pub macro mac() {
        let value = Pub::method::<Priv>;
        //~^ ERROR type `priv_substs::Priv` is private
        value;
        //~^ ERROR type `priv_substs::Priv` is private
        Pub.method::<Priv>();
        //~^ ERROR type `priv_substs::Priv` is private
    }
}
fn priv_substs() {
    priv_substs::mac!();
}

mod priv_parent_substs {
    struct Priv;
    pub struct Pub<T = Priv>(T);
    impl Pub<Priv> {
        pub fn method(&self) {}
        pub fn static_method() {}
        pub const CONST: u8 = 0;
        // pub type AssocTy = u8;
    }

    pub macro mac() {
        let value = <Pub>::method;
        //~^ ERROR type `priv_parent_substs::Priv` is private
        value;
        //~^ ERROR type `priv_parent_substs::Priv` is private
        let value = Pub::method;
        //~^ ERROR type `priv_parent_substs::Priv` is private
        value;
        //~^ ERROR type `priv_parent_substs::Priv` is private
        let value = <Pub>::static_method;
        //~^ ERROR type `priv_parent_substs::Priv` is private
        value;
        //~^ ERROR type `priv_parent_substs::Priv` is private
        let value = Pub::static_method;
        //~^ ERROR type `priv_parent_substs::Priv` is private
        value;
        //~^ ERROR type `priv_parent_substs::Priv` is private
        Pub(Priv).method();
        //~^ ERROR type `priv_parent_substs::Priv` is private

        <Pub>::CONST;
        //~^ ERROR type `priv_parent_substs::Priv` is private
        Pub::CONST;
        //~^ ERROR type `priv_parent_substs::Priv` is private

        // let _: Pub::AssocTy;
        // pub type InSignatureTy = Pub::AssocTy;
    }
}
fn priv_parent_substs() {
    priv_parent_substs::mac!();
}

fn main() {}
