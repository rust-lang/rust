pub struct P;
pub struct Q;
pub struct R<T>(T);

// returns test
pub fn alef() -> &'static [R<P>] { loop {} }
pub fn bet() -> R<[Q; 32]> { loop {} }

// in_args test
pub fn alpha(_x: R<&'static [P]>) { loop {} }
pub fn beta(_x: [R<Q>; 32]) { loop {} }

pub trait TraitCat {}
pub trait TraitDog {}

pub fn gamma<T: TraitCat + TraitDog>(t: [T; 32]) {}

pub fn epsilon<T: TraitCat + TraitDog>(t: &[T]) {}
