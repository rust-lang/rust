pub mod X {
    pub enum A {}
    pub enum B {}
    pub enum C {}
}

pub mod Y {
    pub enum A {}
    pub enum B {}
}

// @has fn_param_ambiguities/fn.f.html //pre 'pub fn f(xa: X::A, ya: Y::A, yb: B, xc: C)'
pub fn f(xa: X::A, ya: Y::A, yb : Y::B, xc: X::C) {}
