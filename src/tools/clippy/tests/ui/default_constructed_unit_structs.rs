#![allow(unused)]
#![warn(clippy::default_constructed_unit_structs)]
use std::marker::PhantomData;

#[derive(Default)]
struct UnitStruct;

impl UnitStruct {
    fn new() -> Self {
        //should lint
        Self::default()
        //~^ default_constructed_unit_structs
    }
}

#[derive(Default)]
struct TupleStruct(usize);

impl TupleStruct {
    fn new() -> Self {
        // should not lint
        Self(Default::default())
    }
}

// no lint for derived impl
#[derive(Default)]
struct NormalStruct {
    inner: PhantomData<usize>,
}

struct NonDefaultStruct;

impl NonDefaultStruct {
    fn default() -> Self {
        Self
    }
}

#[derive(Default)]
enum SomeEnum {
    #[default]
    Unit,
    Tuple(UnitStruct),
    Struct {
        inner: usize,
    },
}

impl NormalStruct {
    fn new() -> Self {
        // should lint
        Self {
            inner: PhantomData::default(),
            //~^ default_constructed_unit_structs
        }
    }

    fn new2() -> Self {
        // should not lint
        Self {
            inner: Default::default(),
        }
    }
}

#[derive(Default)]
struct GenericStruct<T> {
    t: T,
}

impl<T: Default> GenericStruct<T> {
    fn new() -> Self {
        // should not lint
        Self { t: T::default() }
    }

    fn new2() -> Self {
        // should not lint
        Self { t: Default::default() }
    }
}

struct FakeDefault;
impl FakeDefault {
    fn default() -> Self {
        Self
    }
}

impl Default for FakeDefault {
    fn default() -> Self {
        Self
    }
}

#[derive(Default)]
struct EmptyStruct {}

#[derive(Default)]
#[non_exhaustive]
struct NonExhaustiveStruct;

mod issue_10755 {
    struct Sqlite {}

    trait HasArguments<'q> {
        type Arguments;
    }

    impl<'q> HasArguments<'q> for Sqlite {
        type Arguments = std::marker::PhantomData<&'q ()>;
    }

    type SqliteArguments<'q> = <Sqlite as HasArguments<'q>>::Arguments;

    fn foo() {
        // should not lint
        // type alias cannot be used as a constructor
        let _ = <Sqlite as HasArguments>::Arguments::default();

        let _ = SqliteArguments::default();
    }
}

fn main() {
    // should lint
    let _ = PhantomData::<usize>::default();
    //~^ default_constructed_unit_structs
    let _: PhantomData<i32> = PhantomData::default();
    //~^ default_constructed_unit_structs
    let _: PhantomData<i32> = std::marker::PhantomData::default();
    //~^ default_constructed_unit_structs
    let _ = UnitStruct::default();
    //~^ default_constructed_unit_structs

    // should not lint
    let _ = TupleStruct::default();
    let _ = NormalStruct::default();
    let _ = NonExhaustiveStruct::default();
    let _ = SomeEnum::default();
    let _ = NonDefaultStruct::default();
    let _ = EmptyStruct::default();
    let _ = FakeDefault::default();
    let _ = <FakeDefault as Default>::default();

    macro_rules! in_macro {
        ($i:ident) => {{
            let _ = UnitStruct::default();
            let _ = $i::default();
        }};
    }

    in_macro!(UnitStruct);

    macro_rules! struct_from_macro {
        () => {
            UnitStruct
        };
    }

    let _ = <struct_from_macro!()>::default();
}

fn issue12654() {
    #[derive(Default)]
    struct G;

    fn f(_g: G) {}

    f(<_>::default());
    f(<G>::default());
    //~^ default_constructed_unit_structs

    // No lint because `as Default` hides the singleton
    f(<G as Default>::default());
}
