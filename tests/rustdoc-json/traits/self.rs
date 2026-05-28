pub struct Foo;

// Check that Self is represented uniformly between inherent impls, trait impls,
// and trait definitions, even though it uses both SelfTyParam and SelfTyAlias
// internally.
//
// Each assertion matches 3 times, and should be the same each time.

impl Foo {
    //@ ismany '$.index[?(@.name=="by_ref")].inner.function.sig.inputs[0][0]' '"self"' '"self"' '"self"'
    //@ ismany '$.index[?(@.name=="by_ref")].inner.function.sig.inputs[0][1].borrowed_ref.type.generic' '"Self"' '"Self"' '"Self"'
    //@ ismany '$.index[?(@.name=="by_ref")].inner.function.sig.inputs[0][1].borrowed_ref.lifetime' null null null
    //@ ismany '$.index[?(@.name=="by_ref")].inner.function.sig.inputs[0][1].borrowed_ref.is_mutable' false false false
    pub fn by_ref(&self) {}

    //@ ismany '$.index[?(@.name=="by_exclusive_ref")].inner.function.sig.inputs[0][0]' '"self"' '"self"' '"self"'
    //@ ismany '$.index[?(@.name=="by_exclusive_ref")].inner.function.sig.inputs[0][1].borrowed_ref.type.generic' '"Self"' '"Self"' '"Self"'
    //@ ismany '$.index[?(@.name=="by_exclusive_ref")].inner.function.sig.inputs[0][1].borrowed_ref.lifetime' null null null
    //@ ismany '$.index[?(@.name=="by_exclusive_ref")].inner.function.sig.inputs[0][1].borrowed_ref.is_mutable' true true true
    pub fn by_exclusive_ref(&mut self) {}

    //@ ismany '$.index[?(@.name=="by_value")].inner.function.sig.inputs[0][0]' '"self"' '"self"' '"self"'
    //@ ismany '$.index[?(@.name=="by_value")].inner.function.sig.inputs[0][1].generic' '"Self"' '"Self"' '"Self"'
    pub fn by_value(self) {}

    //@ ismany '$.index[?(@.name=="with_lifetime")].inner.function.sig.inputs[0][0]' '"self"' '"self"' '"self"'
    //@ ismany '$.index[?(@.name=="with_lifetime")].inner.function.sig.inputs[0][1].borrowed_ref.type.generic' '"Self"' '"Self"' '"Self"'
    //@ ismany '$.index[?(@.name=="with_lifetime")].inner.function.sig.inputs[0][1].borrowed_ref.lifetime' \"\'a\" \"\'a\" \"\'a\"
    //@ ismany '$.index[?(@.name=="with_lifetime")].inner.function.sig.inputs[0][1].borrowed_ref.is_mutable' false false false
    pub fn with_lifetime<'a>(&'a self) {}

    //@ ismany '$.index[?(@.name=="build")].inner.function.sig.output.generic' '"Self"' '"Self"' '"Self"'
    pub fn build() -> Self {
        Self
    }
}

pub struct Bar;

pub trait SelfParams {
    fn by_ref(&self);
    fn by_exclusive_ref(&mut self);
    fn by_value(self);
    fn with_lifetime<'a>(&'a self);
    fn build() -> Self;
}

impl SelfParams for Bar {
    fn by_ref(&self) {}
    fn by_exclusive_ref(&mut self) {}
    fn by_value(self) {}
    fn with_lifetime<'a>(&'a self) {}
    fn build() -> Self {
        Self
    }
}
