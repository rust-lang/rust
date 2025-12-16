#![feature(auto_traits, negative_impls)]

pub auto trait Marker {}

// pub struct Outer<T>(Inner<T>);
// struct Inner<T>(T);

// pub struct Compound {
//     f0: i32,
//     f1: u32,
// }

// pub struct Compound<T> {
//     f0: i32,
//     f1: Wrap<T>,
//     // f1: T,
// }

// pub struct Wrap<T>(T);

// impl<T: Copy> Marker for Wrap<T> {}

// [[next]] no explicit impl found
// [[next]]  (G) Err(NoSolution)  Misc  #C=1  for[] TraitPredicate(<Compound<T> as Marker>, polarity:Positive)
// [[next]]   (C#0) Err(NoSolution)  TraitCandidate/BuiltinImpl(Misc)
// [[next]]    (G) Ok(Yes)  ImplWhereBound  #C=1  for[] TraitPredicate(<i32 as Marker>, polarity:Positive)
// [[next]]     (C#0) Ok(Yes)  TraitCandidate/BuiltinImpl(Misc)
// [[next]]    (G) Err(NoSolution)  ImplWhereBound  #C=1  for[] TraitPredicate(<Wrap<T> as Marker>, polarity:Positive)
// [[next]]     (C#0) Err(NoSolution)  TraitCandidate/Impl(DefId(0:12 ~ t[6e8d]::{impl#0}))
// [[next]]      (G) Ok(Yes)  ImplWhereBound  #C=1  for[] TraitPredicate(<T as std::marker::Sized>, polarity:Positive)
// [[next]]       (C#0) Ok(Yes)  TraitCandidate/ParamEnv(NonGlobal)
// [[next]]      (G) Err(NoSolution)  ImplWhereBound  #C=0  for[] TraitPredicate(<T as std::marker::Copy>, polarity:Positive)

// pub struct Compound {
//     f0: NonMarker,
// }

// struct NonMarker;
// impl !Marker for NonMarker {}

//////////7

// pub struct Outer<T>(Inner<T>);
// struct Inner<T>(T);

// impl<T, U> Marker for Inner<T>
// where
//     T: Iterator<Item = U>,
//     U: Copy,
// {}

// FIXME: Negative
// impl<T> Marker for Inner<T>
// where
// //     // NOTE: Can be arbitrarily deeply nested
//     // T: Iterator<Item: Copy>,
    // T: Iterator<Item: Iterator<Item = ()>>
//  T: Iterator<Item = ()>,
// {}

//
// impl<T, X> Marker for Inner<T>
// where
//     T: Iterator<Item = X>,
//     (X,): Copy, // FIXME: dropped
// {}

// impl<'a, T, const N: usize> Marker for Inner<T>
// where
//     T: Iterator<Item = [&'a (); N]>,
// {}

// impl<T: !Copy> Marker for Inner<T> {}

// impl<T> Marker for Inner<T>
// where
//     T: const std::default::Default,
// {}

// impl<T> const Marker for Inner<T>
// where
//     T: [const] std::default::Default,
// {}

// impl<T> !Marker for Inner<T> {}
// impl<T> Marker for Inner<T>
// where
//     // T: Copy
//     [T; 1 + 2]:
// {}

// pub trait Bound {}

// impl<T> Bound for T {}

// impl<X, Y> Marker for Inner<(X, Y)> {}

// impl<T: Iterator<Item = String>> Marker for Inner<T> {}
// impl<T: Iterator<Item: Copy>> Marker for Inner<T> {}

// pub trait Bound {}
// impl<T: Copy + std::fmt::Debug + Bound> Marker for Inner<T> {}
// impl Marker for Inner<&'static str> {}

///////////// for<>fnptr-ICE //////////////

// pub struct Outer<'a>(Local<fn(&'a ())>);

// struct Local<T>(T);

// impl Marker for Local<for<'a> fn(&'a ())> {}

///////////

// pub struct Outer<'a>(Inner<'a>);
// struct Inner<'a>(&'a ());

// impl Marker for Inner<'static> {}

///////////////// ConstEvaluatable

// pub struct Hold<const N: u8>;

// pub struct Outer<const N: u8>(Inner<N>);
// struct Inner<const N: u8>;

// impl<const N: u8> Marker for Inner<N>
// where
//     Hold<{ N + 1 }>:
// {}

// pub struct Type<T>(Opaque<T>);

// type Opaque<T> = impl Sized;

// #[define_opaque(Opaque)] fn define<T>(x: T) -> Opaque<T> { Wrap(x) }

// pub struct Wrap<T>(T);

// impl<T: Copy> Marker for Wrap<T> {}

// impl !Marker for () {}
//

//////// XXX outlives-preds (BUG!)

// pub struct Outer<'a, T: 'a>(Inner<'a, T>);
// struct Inner<'a, T: 'a>(&'a T);

// impl<'a, T> Marker for Inner<'a, T>
// where
//     'a: 'static,
//     // T: Bound<'a>, // [!]
// {}

// pub trait Bound<'a> {}

///////////

pub struct Outer<T>(Inner<T>);
type Inner<T> = impl Sized; #[define_opaque(Inner)] fn define<T>() -> Inner<T> {}

