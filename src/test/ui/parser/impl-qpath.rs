// build-pass (FIXME(62277): could be check-pass?)
// compile-flags: -Z parse-only

impl <*const u8>::AssocTy {} // OK
impl <Type as Trait>::AssocTy {} // OK
impl <'a + Trait>::AssocTy {} // OK
impl <<Type>::AssocTy>::AssocTy {} // OK
