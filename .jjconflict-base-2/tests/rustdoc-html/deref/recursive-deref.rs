use std::ops::Deref;

// Cyclic deref with the parent (which is not the top parent).
pub struct A;
pub struct B;
pub struct C;

impl C {
    pub fn c(&self) {}
}

//@ has recursive_deref/struct.A.html '//h3[@class="code-header"]' 'impl Deref for A'
//@ has '-' '//*[@class="impl-items"]//*[@id="method.c"]' 'pub fn c(&self)'
impl Deref for A {
    type Target = B;

    fn deref(&self) -> &Self::Target {
        panic!()
    }
}

//@ has recursive_deref/struct.B.html '//h3[@class="code-header"]' 'impl Deref for B'
//@ has '-' '//*[@class="impl-items"]//*[@id="method.c"]' 'pub fn c(&self)'
impl Deref for B {
    type Target = C;

    fn deref(&self) -> &Self::Target {
        panic!()
    }
}

//@ has recursive_deref/struct.C.html '//h3[@class="code-header"]' 'impl Deref for C'
impl Deref for C {
    type Target = B;

    fn deref(&self) -> &Self::Target {
        panic!()
    }
}

// Cyclic deref with the grand-parent (which is not the top parent).
pub struct D;
pub struct E;
pub struct F;
pub struct G;

impl G {
    // There is no "self" parameter so it shouldn't be listed!
    pub fn g() {}
}

//@ has recursive_deref/struct.D.html '//h3[@class="code-header"]' 'impl Deref for D'
// We also check that `G::g` method isn't rendered because there is no `self` argument.
//@ !has '-' '//*[@id="deref-methods-G"]' ''
impl Deref for D {
    type Target = E;

    fn deref(&self) -> &Self::Target {
        panic!()
    }
}

//@ has recursive_deref/struct.E.html '//h3[@class="code-header"]' 'impl Deref for E'
// We also check that `G::g` method isn't rendered because there is no `self` argument.
//@ !has '-' '//*[@id="deref-methods-G"]' ''
impl Deref for E {
    type Target = F;

    fn deref(&self) -> &Self::Target {
        panic!()
    }
}

//@ has recursive_deref/struct.F.html '//h3[@class="code-header"]' 'impl Deref for F'
// We also check that `G::g` method isn't rendered because there is no `self` argument.
//@ !has '-' '//*[@id="deref-methods-G"]' ''
impl Deref for F {
    type Target = G;

    fn deref(&self) -> &Self::Target {
        panic!()
    }
}

//@ has recursive_deref/struct.G.html '//h3[@class="code-header"]' 'impl Deref for G'
impl Deref for G {
    type Target = E;

    fn deref(&self) -> &Self::Target {
        panic!()
    }
}

// Cyclic deref with top parent.
pub struct H;
pub struct I;

impl I {
    // There is no "self" parameter so it shouldn't be listed!
    pub fn i() {}
}

//@ has recursive_deref/struct.H.html '//h3[@class="code-header"]' 'impl Deref for H'
//@ !has '-' '//*[@id="deref-methods-I"]' ''
impl Deref for H {
    type Target = I;

    fn deref(&self) -> &Self::Target {
        panic!()
    }
}

//@ has recursive_deref/struct.I.html '//h3[@class="code-header"]' 'impl Deref for I'
impl Deref for I {
    type Target = H;

    fn deref(&self) -> &Self::Target {
        panic!()
    }
}
