pub struct Foo;

// Check that Self is represented uniformly between inherent impls, trait impls,
// and trait definitions, even though it uses both SelfTyParam and SelfTyAlias
// internally.
//
// Each assertion matches 3 times, and should be the same each time.

impl Foo {
    //@ ismany '$.index[?(@.name=="by_ref")].inner.function.sig.inputs[0][0]' '"self"' '"self"' '"self"'
    //@ ismany '$.index[?(@.name=="by_ref")].inner.function.sig.inputs[0][1]' 1 1 1
    //@ is '$.types[1].borrowed_ref.type' 0
    //@ is '$.types[0].generic' '"Self"'
    //@ is '$.types[1].borrowed_ref.lifetime' null
    //@ is '$.types[1].borrowed_ref.is_mutable' false
    pub fn by_ref(&self) {}

    //@ ismany '$.index[?(@.name=="by_exclusive_ref")].inner.function.sig.inputs[0][0]' '"self"' '"self"' '"self"'
    //@ ismany '$.index[?(@.name=="by_exclusive_ref")].inner.function.sig.inputs[0][1]' 2 2 2
    //@ is '$.types[2].borrowed_ref.type' 0
    //@ is '$.types[2].borrowed_ref.lifetime' null
    //@ is '$.types[2].borrowed_ref.is_mutable' true
    pub fn by_exclusive_ref(&mut self) {}

    //@ ismany '$.index[?(@.name=="by_value")].inner.function.sig.inputs[0][0]' '"self"' '"self"' '"self"'
    //@ ismany '$.index[?(@.name=="by_value")].inner.function.sig.inputs[0][1]' 0 0 0
    pub fn by_value(self) {}

    //@ ismany '$.index[?(@.name=="with_lifetime")].inner.function.sig.inputs[0][0]' '"self"' '"self"' '"self"'
    //@ ismany '$.index[?(@.name=="with_lifetime")].inner.function.sig.inputs[0][1]' 3 3 3
    //@ is '$.types[3].borrowed_ref.type' 0
    //@ is '$.types[3].borrowed_ref.lifetime' \"\'a\"
    //@ is '$.types[3].borrowed_ref.is_mutable' false
    pub fn with_lifetime<'a>(&'a self) {}

    //@ ismany '$.index[?(@.name=="build")].inner.function.sig.output' 0 0 0
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
