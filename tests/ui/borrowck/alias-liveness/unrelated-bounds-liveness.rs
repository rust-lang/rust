//@ edition: 2024

// This test shows that aliases can end up in the hidden type of an impl Trait.

#![forbid(unsafe_code)]

use std::fmt::Debug;

trait Tr {
    type Assoc: Debug;
    fn get(self) -> Self::Assoc;
}
impl<'q> Tr for &'q u8 {
    type Assoc = &'q u8;
    fn get(self) -> &'q u8 {
        self
    }
}

fn make<'a, 'b, T: Tr>(t: T) -> impl Debug + 'a + 'b
where
    T::Assoc: 'a + 'b,
{
    t.get()
}

fn show<T: Debug>(t: T, r: &Vec<u8>) {
    println!("escaped = {t:?}, reused = {r:?}");
}

fn base() {
    let escaped;
    {
        let v: Vec<u8> = vec![42];
        escaped = make(&v[0]);
        //~^ ERROR `v` does not live long enough
    }
    let reused: Vec<u8> = vec![7];
    show(escaped, &reused);
}

mod item_bounds {
    use super::{Debug, show};

    trait Tr<'a, 'b> {
        type Assoc: Debug + 'a + 'b;
        fn get(self) -> Self::Assoc;
    }

    impl<'a, 'b, 'q: 'a + 'b> Tr<'a, 'b> for &'q u8 {
        type Assoc = &'q u8;
        fn get(self) -> &'q u8 {
            self
        }
    }

    fn make<'a, 'b, T: Tr<'a, 'b>>(t: T) -> impl Debug + 'a + 'b {
        t.get()
    }

    pub fn run() {
        let escaped;
        {
            let v: Vec<u8> = vec![42];
            escaped = make(&v[0]);
            //~^ ERROR `v` does not live long enough
        }
        let reused: Vec<u8> = vec![7];
        show(escaped, &reused);
    }
}

// Still unsound.
mod lifetime_arg {
    use super::{Debug, show};

    mod trait_arg {
        use super::Debug;

        pub trait Tr<'c> {
            type Assoc: Debug;
            fn get(self, x: &'c u8) -> Self::Assoc;
        }

        impl<'c> Tr<'c> for () {
            type Assoc = &'c u8;
            fn get(self, x: &'c u8) -> &'c u8 {
                x
            }
        }

        pub fn make<'a, 'b, 'c, T: Tr<'c>>(t: T, x: &'c u8) -> impl Debug + 'a + 'b
        where
            <T as Tr<'c>>::Assoc: 'a + 'b,
        {
            t.get(x)
        }
    }

    mod gat_arg {
        use super::Debug;

        pub trait Tr {
            type Assoc<'z>: Debug;
            fn get<'z>(self, x: &'z u8) -> Self::Assoc<'z>;
        }

        impl Tr for () {
            type Assoc<'z> = &'z u8;
            fn get<'z>(self, x: &'z u8) -> &'z u8 {
                x
            }
        }

        pub fn make<'a, 'b, 'z, T: Tr>(t: T, x: &'z u8) -> impl Debug + 'a + 'b
        where
            T::Assoc<'z>: 'a + 'b,
        {
            t.get(x)
        }
    }

    pub fn on_the_trait() {
        let escaped;
        {
            let v: Vec<u8> = vec![42];
            escaped = trait_arg::make((), &v[0]);
            //~^ ERROR `v` does not live long enough
        }
        let reused: Vec<u8> = vec![7];
        show(escaped, &reused);
    }

    pub fn on_the_gat() {
        let escaped;
        {
            let v: Vec<u8> = vec![42];
            escaped = gat_arg::make((), &v[0]);
            //~^ ERROR `v` does not live long enough
        }
        let reused: Vec<u8> = vec![7];
        show(escaped, &reused);
    }
}

mod nested {
    use super::{Debug, Tr, show};

    fn inner<'a, 'b, T: Tr>(t: T) -> impl Debug + 'a + 'b
    where
        T::Assoc: 'a + 'b,
    {
        t.get()
    }

    fn outer<'a, 'b, T: Tr>(t: T) -> impl Debug + 'a + 'b
    where
        T::Assoc: 'a + 'b,
    {
        inner::<'a, 'b, T>(t)
    }

    pub fn run() {
        let escaped;
        {
            let v: Vec<u8> = vec![42];
            escaped = outer(&v[0]);
            //~^ ERROR `v` does not live long enough
        }
        let reused: Vec<u8> = vec![7];
        show(escaped, &reused);
    }
}

mod rpitit {
    use super::{Debug, Tr, show};

    trait Make<'a, 'b> {
        fn make<T: Tr>(t: T) -> impl Debug + 'a + 'b
        where
            T::Assoc: 'a + 'b;
    }

    impl<'a, 'b> Make<'a, 'b> for () {
        fn make<T: Tr>(t: T) -> impl Debug + 'a + 'b
        where
            T::Assoc: 'a + 'b,
        {
            t.get()
        }
    }

    pub fn run() {
        let escaped;
        {
            let v: Vec<u8> = vec![42];
            escaped = <() as Make>::make(&v[0]);
            //~^ ERROR `v` does not live long enough
        }
        let reused: Vec<u8> = vec![7];
        show(escaped, &reused);
    }
}

mod three_bounds {
    use super::{Debug, Tr, show};

    fn make<'a, 'b, 'c: 'a, T: Tr>(t: T) -> impl Debug + 'a + 'b + 'c
    where
        T::Assoc: 'a + 'b + 'c,
    {
        t.get()
    }

    pub fn run() {
        let escaped;
        {
            let v: Vec<u8> = vec![42];
            escaped = make(&v[0]);
            //~^ ERROR `v` does not live long enough
        }
        let reused: Vec<u8> = vec![7];
        show(escaped, &reused);
    }
}

// If we have a 'static bound for an associated type, in theory that should be
// enough that the associated type can be used within an `impl Trait` without
// the self type needing to be live. In practice, this precision is tricky,
// because to do this correctly, we would essentially need to ask "is there any
// projection from a given type parameter that can end up in the hidden type of
// an opaque type?", which is *hard*.
mod static_associated_type {
    use super::{Debug, show};

    trait Tr {
        type Assoc: Debug;
        fn get(self) -> Self::Assoc;
    }
    impl<'q> Tr for &'q u8 {
        type Assoc = &'static str;
        fn get(self) -> &'static str {
            "hi"
        }
    }

    fn make<'a, 'b, T: Tr>(t: T) -> impl Debug + 'a + 'b
    where
        T::Assoc: 'static,
    {
        t.get()
    }

    pub fn run() {
        let escaped;
        {
            let v: Vec<u8> = vec![42];
            escaped = make(&v[0]);
            //~^ ERROR `v` does not live long enough
        }
        let reused: Vec<u8> = vec![7];
        show(escaped, &reused);
    }
}

// If the self type has only *some* lifetimes can can be in the associated type,
// then in theory we don't need to keep the other lifetimes live.
// In practice, this precision is tricky,
mod one_lifetime {
    use super::{Debug, Tr, make, show};

    impl<'q, 'r> Tr for (&'q u8, &'r u8) {
        type Assoc = &'q u8;
        fn get(self) -> &'q u8 {
            self.0
        }
    }

    pub fn run() {
        let keep: Vec<u8> = vec![1];
        let escaped;
        {
            let v: Vec<u8> = vec![42];
            escaped = make((&keep[0], &v[0]));
            //~^ ERROR `v` does not live long enough
        }
        let reused: Vec<u8> = vec![7];
        show(escaped, &reused);
    }

}

// The function signature for the RPIT can in theory be used to provide implied
// bounds as to the lifetimes that can be contained within an associated type.
mod implied_bounds {
    use std::marker::PhantomData;

    use super::{Debug, Tr, show};

    fn make<'a, 'b, T: Tr>(
        t: T,
        _s: PhantomData<&'a Option<T::Assoc>>,
        _u: PhantomData<&'b Option<T::Assoc>>,
    ) -> impl Debug + 'a + 'b {
        t.get()
    }

    pub fn run() {
        let escaped;
        {
            let v: Vec<u8> = vec![42];
            escaped = make(&v[0], PhantomData, PhantomData);
            //~^ ERROR `v` does not live long enough
        }
        let reused: Vec<u8> = vec![7];
        show(escaped, &reused);
    }
}

fn main() {
    base();
    item_bounds::run();
    lifetime_arg::on_the_trait();
    lifetime_arg::on_the_gat();
    nested::run();
    rpitit::run();
    three_bounds::run();
    static_associated_type::run();
    one_lifetime::run();
    implied_bounds::run();
}
