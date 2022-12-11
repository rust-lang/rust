// Same as ./compare-impl-method.rs but we shouldn't use implied bounds from projections in impl
// header. See `mod exploit` for why.
// check-fail

trait Project {
    type Ty;
}
impl<T> Project for T {
    type Ty = ();
}

trait Trait {
    fn get();
}

impl<'a, 'b> Trait for <&'a &'b () as Project>::Ty {
    fn get()
    where
        'b: 'a,
        //~^ ERROR impl has stricter requirements than trait
    {
    }
}

mod exploit {
    trait Trait<Witness> {
        fn extend(self) -> &'static str;
    }

    impl<'a> Trait<<&'static &'a u8 as super::Project>::Ty> for &'a str {
        fn extend(self) -> &'static str
        where
            'a: 'static,
            //~^ ERROR impl has stricter requirements than trait
            //~| WARN unnecessary lifetime
        {
            self
        }
    }

    fn main() {
        let val = <&str as Trait<()>>::extend(&String::from("blah blah blah"));
        println!("{}", val);
    }
}

fn main() {}
