pub trait DeclaredTrait {
    type Type;
}

impl DeclaredTrait for i32 {
    type Type = i32;
}

pub trait WhereTrait {
    type Type;
}

impl WhereTrait for i32 {
    type Type = i32;
}

// Make sure we don't add a bound that just shares a name with an associated
// type.
pub mod module {
    pub type Type = i32;
}

#[derive(PartialEq, Debug)]
struct PrivateStruct<T>(T);

#[derive(PartialEq, Debug)]
struct TupleStruct<A, B: DeclaredTrait, C>(
    module::Type,
    Option<module::Type>,
    A,
    PrivateStruct<A>,
    B,
    B::Type,
    Option<B::Type>,
    <B as DeclaredTrait>::Type,
    Option<<B as DeclaredTrait>::Type>,
    C,
    C::Type,
    Option<C::Type>,
    <C as WhereTrait>::Type,
    Option<<C as WhereTrait>::Type>,
    <i32 as DeclaredTrait>::Type,
)
where
    C: WhereTrait;

#[derive(PartialEq, Debug)]
pub struct Struct<A, B: DeclaredTrait, C>
where
    C: WhereTrait,
{
    m1: module::Type,
    m2: Option<module::Type>,
    a1: A,
    a2: PrivateStruct<A>,
    b: B,
    b1: B::Type,
    b2: Option<B::Type>,
    b3: <B as DeclaredTrait>::Type,
    b4: Option<<B as DeclaredTrait>::Type>,
    c: C,
    c1: C::Type,
    c2: Option<C::Type>,
    c3: <C as WhereTrait>::Type,
    c4: Option<<C as WhereTrait>::Type>,
    d: <i32 as DeclaredTrait>::Type,
}

#[derive(PartialEq, Debug)]
enum Enum<A, B: DeclaredTrait, C>
where
    C: WhereTrait,
{
    Unit,
    Seq(
        module::Type,
        Option<module::Type>,
        A,
        PrivateStruct<A>,
        B,
        B::Type,
        Option<B::Type>,
        <B as DeclaredTrait>::Type,
        Option<<B as DeclaredTrait>::Type>,
        C,
        C::Type,
        Option<C::Type>,
        <C as WhereTrait>::Type,
        Option<<C as WhereTrait>::Type>,
        <i32 as DeclaredTrait>::Type,
    ),
    Map {
        m1: module::Type,
        m2: Option<module::Type>,
        a1: A,
        a2: PrivateStruct<A>,
        b: B,
        b1: B::Type,
        b2: Option<B::Type>,
        b3: <B as DeclaredTrait>::Type,
        b4: Option<<B as DeclaredTrait>::Type>,
        c: C,
        c1: C::Type,
        c2: Option<C::Type>,
        c3: <C as WhereTrait>::Type,
        c4: Option<<C as WhereTrait>::Type>,
        d: <i32 as DeclaredTrait>::Type,
    },
}

fn main() {
    let e: Enum<i32, i32, i32> =
        Enum::Seq(0, None, 0, PrivateStruct(0), 0, 0, None, 0, None, 0, 0, None, 0, None, 0);
    assert_eq!(e, e);

    let e: Enum<i32, i32, i32> = Enum::Map {
        m1: 0,
        m2: None,
        a1: 0,
        a2: PrivateStruct(0),
        b: 0,
        b1: 0,
        b2: None,
        b3: 0,
        b4: None,
        c: 0,
        c1: 0,
        c2: None,
        c3: 0,
        c4: None,
        d: 0,
    };
    assert_eq!(e, e);
    let e: TupleStruct<i32, i32, i32> =
        TupleStruct(0, None, 0, PrivateStruct(0), 0, 0, None, 0, None, 0, 0, None, 0, None, 0);
    assert_eq!(e, e);

    let e: Struct<i32, i32, i32> = Struct {
        m1: 0,
        m2: None,
        a1: 0,
        a2: PrivateStruct(0),
        b: 0,
        b1: 0,
        b2: None,
        b3: 0,
        b4: None,
        c: 0,
        c1: 0,
        c2: None,
        c3: 0,
        c4: None,
        d: 0,
    };
    assert_eq!(e, e);

    let e = Enum::Unit::<i32, i32, i32>;
    assert_eq!(e, e);
}
