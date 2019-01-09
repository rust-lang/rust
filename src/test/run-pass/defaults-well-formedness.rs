#![allow(dead_code)]
trait Trait<T> {}
struct Foo<U, V=i32>(U, V) where U: Trait<V>;

trait Marker {}
struct TwoParams<T, U>(T, U);
impl Marker for TwoParams<i32, i32> {}

// Clauses with more than 1 param are not checked.
struct IndividuallyBogus<T = i32, U = i32>(TwoParams<T, U>) where TwoParams<T, U>: Marker;
struct BogusTogether<T = u32, U = i32>(T, U) where TwoParams<T, U>: Marker;
// Clauses with non-defaulted params are not checked.
struct NonDefaultedInClause<T, U = i32>(TwoParams<T, U>) where TwoParams<T, U>: Marker;
struct DefaultedLhs<U, V=i32>(U, V) where V: Trait<U>;
// Dependent defaults are not checked.
struct Dependent<T, U = T>(T, U) where U: Copy;
trait SelfBound<T: Copy=Self> {}
// Not even for well-formedness.
struct WellFormedProjection<A, T=<A as Iterator>::Item>(A, T);

// Issue #49344, predicates with lifetimes should not be checked.
trait Scope<'a> {}
struct Request<'a, S: Scope<'a> = i32>(S, &'a ());

fn main() {}
