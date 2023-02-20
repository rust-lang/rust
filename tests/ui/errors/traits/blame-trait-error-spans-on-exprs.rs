// This test examines the error spans reported when a generic `impl` fails.
// For example, if a function wants an `Option<T>` where `T: Copy` but you pass `Some(vec![1, 2])`,
// then we want to point at the `vec![1, 2]` and not the `Some( ... )` expression.

trait T1 {}
trait T2 {}
trait T3 {}
trait T4 {}

impl T2 for i32 {}
impl T3 for i32 {}

struct Wrapper<W> {
    value: W,
}
impl<B: T2> T1 for Wrapper<B> {}

struct Burrito<F> {
    spicy: bool,
    filling: F,
}
impl<A: T3> T2 for Burrito<A> {}

struct BurritoTuple<F>(F);
impl<C: T3> T2 for BurritoTuple<C> {}

enum BurritoKinds<G> {
    SmallBurrito { spicy: bool, small_filling: G },
    LargeBurrito { spicy: bool, large_filling: G },
    MultiBurrito { first_filling: G, second_filling: G },
}
impl<D: T3> T2 for BurritoKinds<D> {}

struct Taco<H>(bool, H);
impl<E: T3> T2 for Taco<E> {}

enum TacoKinds<H> {
    OneTaco(bool, H),
    TwoTacos(bool, H, H),
}
impl<F: T3> T2 for TacoKinds<F> {}

struct GenericBurrito<Spiciness, Filling> {
    spiciness: Spiciness,
    filling: Filling,
}
impl<X, Y: T3> T2 for GenericBurrito<X, Y> {}
struct NotSpicy;

impl<A: T3, B: T3> T2 for (A, B) {}
impl<A: T2, B: T2> T1 for (A, B) {}

fn want<V: T1>(_x: V) {}

// Some more-complex examples:
type AliasBurrito<T> = GenericBurrito<T, T>;

// The following example is fairly confusing. The idea is that we want to "misdirect" the location
// of the error.

struct Two<A, B> {
    a: A,
    b: B,
}

impl<X, Y: T1, Z> T1 for Two<Two<X, Y>, Z> {}

struct DoubleWrapper<T> {
    item: Wrapper<T>,
}

impl<T: T1> T1 for DoubleWrapper<T> {}

impl<'a, T: T2> T1 for &'a T {}

fn example<Q>(q: Q) {
    // In each of the following examples, we expect the error span to point at the 'q' variable,
    // since the missing constraint is `Q: T3`.

    // Verifies for struct:
    want(Wrapper { value: Burrito { spicy: false, filling: q } });
    //~^ ERROR the trait bound `Q: T3` is not satisfied [E0277]

    // Verifies for enum with named fields in variant:
    want(Wrapper { value: BurritoKinds::SmallBurrito { spicy: true, small_filling: q } });
    //~^ ERROR the trait bound `Q: T3` is not satisfied [E0277]

    // Verifies for tuple struct:
    want(Wrapper { value: Taco(false, q) });
    //~^ ERROR the trait bound `Q: T3` is not satisfied [E0277]

    // Verifies for tuple enum variant:
    want(Wrapper { value: TacoKinds::OneTaco(false, q) });
    //~^ ERROR the trait bound `Q: T3` is not satisfied [E0277]

    // Verifies for generic type with multiple parameters:
    want(Wrapper { value: GenericBurrito { spiciness: NotSpicy, filling: q } });
    //~^ ERROR the trait bound `Q: T3` is not satisfied [E0277]

    // Verifies for tuple:
    want((3, q));
    //~^ ERROR the trait bound `Q: T2` is not satisfied [E0277]

    // Verifies for nested tuple:
    want(Wrapper { value: (3, q) });
    //~^ ERROR the trait bound `Q: T3` is not satisfied [E0277]

    // Verifies for nested tuple:
    want(((3, q), 5));
    //~^ ERROR the trait bound `Q: T3` is not satisfied [E0277]

    want(DoubleWrapper { item: Wrapper { value: q } });
    //~^ ERROR the trait bound `Q: T1` is not satisfied [E0277]

    want(DoubleWrapper { item: Wrapper { value: DoubleWrapper { item: Wrapper { value: q } } } });
    //~^ ERROR the trait bound `Q: T1` is not satisfied [E0277]

    // Verifies for type alias to struct:
    want(Wrapper { value: AliasBurrito { spiciness: q, filling: q } });
    //~^ ERROR the trait bound `Q: T3` is not satisfied [E0277]

    want(Two { a: Two { a: (), b: q }, b: () });
    //~^ ERROR the trait bound `Q: T1` is not satisfied [E0277]

    // We *should* blame the 'q'.
    // FIXME: Right now, the wrong field is blamed.
    want(
        Two { a: Two { a: (), b: Two { a: Two { a: (), b: q }, b: () } }, b: () },
        //~^ ERROR the trait bound `Q: T1` is not satisfied [E0277]
    );

    // Verifies for reference:
    want(&Burrito { spicy: false, filling: q });
    //~^ ERROR the trait bound `Q: T3` is not satisfied [E0277]
}

fn main() {}
