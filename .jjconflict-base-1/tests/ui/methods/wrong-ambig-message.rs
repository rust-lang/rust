fn main() {
    trait Hello {
        fn name(&self) -> String;
    }

    #[derive(Debug)]
    struct Container2 {
        val: String,
    }

    trait AName2 {
        fn name(&self) -> String;
    }

    trait BName2 {
        fn name(&self, v: bool) -> String;
    }

    impl AName2 for Container2 {
        fn name(&self) -> String {
            "aname2".into()
        }
    }

    impl BName2 for Container2 {
        fn name(&self, _v: bool) -> String {
            "bname2".into()
        }
    }

    let c2 = Container2 { val: "abc".into() };
    println!("c2 = {:?}", c2.name());
    //~^ ERROR: multiple applicable items in scope
}
