pub struct P;
pub struct Q;
pub struct R<T>(T);

// returns test
pub fn alef() -> R<P> { loop {} }
pub fn bet() -> R<Q> { loop {} }

// in_args test
pub fn alpha(_x: R<P>) { loop {} }
pub fn beta(_x: R<Q>) { loop {} }

// test case with multiple appearances of the same type
pub struct ExtraCreditStructMulti<T, U> { t: T, u: U }
pub struct ExtraCreditInnerMulti {}
pub fn extracreditlabhomework(
    _param: ExtraCreditStructMulti<ExtraCreditInnerMulti, ExtraCreditInnerMulti>
) { loop {} }
pub fn redherringmatchforextracredit(
    _param: ExtraCreditStructMulti<ExtraCreditInnerMulti, ()>
) { loop {} }

pub trait TraitCat {}
pub trait TraitDog {}

pub fn gamma<T: TraitCat + TraitDog>(t: T) {}
