pub type BoxedStr = Box<str>;
pub type IntMap = std::collections::HashMap<i64, u64>;

pub struct TyPair<T, U = T>(T, U);

pub type T0 = TyPair<i32>;
pub type T1 = TyPair<i32, u32>;
pub type T2<K> = TyPair<i32, K>;
pub type T3<Q> = TyPair<Q, Q>;

pub struct CtPair<const C: u32, const D: u32 = C>;

pub type C0 = CtPair<43, 43>;
pub type C1 = CtPair<0, 1>;
pub type C2 = CtPair<{1 + 2}, 3>;

pub struct Re<'a, U = &'a ()>(&'a (), U);

pub type R0<'q> = Re<'q>;
pub type R1<'q> = Re<'q, &'q ()>;
pub type R2<'q> = Re<'q, &'static ()>;
pub type H0 = fn(for<'a> fn(Re<'a>));
pub type H1 = for<'b> fn(for<'a> fn(Re<'a, &'b ()>));
pub type H2 = for<'a> fn(for<'b> fn(Re<'a, &'b ()>));

pub struct Proj<T: Basis, U = <T as Basis>::Assoc>(T, U);
pub trait Basis { type Assoc; }
impl Basis for () { type Assoc = bool; }

pub type P0 = Proj<()>;
pub type P1 = Proj<(), bool>;
pub type P2 = Proj<(), ()>;

pub struct Alpha<T = for<'any> fn(&'any ())>(T);

pub type A0 = Alpha;
pub type A1 = Alpha<for<'arbitrary> fn(&'arbitrary ())>;

pub struct Multi<A = u64, B = u64>(A, B);

pub type M0 = Multi<u64, ()>;

pub trait Trait0<'a, T = &'a ()> {}
pub type D0 = dyn for<'a> Trait0<'a>;

// Regression test for issue #119529.
pub trait Trait1<T = (), const K: u32 = 0> {}
pub type D1<T> = dyn Trait1<T>;
pub type D2<const K: u32> = dyn Trait1<(), K>;
pub type D3 = dyn Trait1;
