// Core operators

#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

#[lang="drop"]
pub trait Drop {
    fn finalize(&self);  // XXX: Rename to "drop"? --pcwalton
}

#[lang="add"]
pub trait Add<RHS,Result> {
    pure fn add(rhs: &RHS) -> Result;
}

#[lang="sub"]
pub trait Sub<RHS,Result> {
    pure fn sub(&self, rhs: &RHS) -> Result;
}

#[lang="mul"]
pub trait Mul<RHS,Result> {
    pure fn mul(&self, rhs: &RHS) -> Result;
}

#[lang="div"]
pub trait Div<RHS,Result> {
    pure fn div(&self, rhs: &RHS) -> Result;
}

#[lang="modulo"]
pub trait Modulo<RHS,Result> {
    pure fn modulo(&self, rhs: &RHS) -> Result;
}

#[lang="neg"]
pub trait Neg<Result> {
    pure fn neg(&self) -> Result;
}

#[lang="bitand"]
pub trait BitAnd<RHS,Result> {
    pure fn bitand(&self, rhs: &RHS) -> Result;
}

#[lang="bitor"]
pub trait BitOr<RHS,Result> {
    pure fn bitor(&self, rhs: &RHS) -> Result;
}

#[lang="bitxor"]
pub trait BitXor<RHS,Result> {
    pure fn bitxor(&self, rhs: &RHS) -> Result;
}

#[lang="shl"]
pub trait Shl<RHS,Result> {
    pure fn shl(&self, rhs: &RHS) -> Result;
}

#[lang="shr"]
pub trait Shr<RHS,Result> {
    pure fn shr(&self, rhs: &RHS) -> Result;
}

#[lang="index"]
pub trait Index<Index,Result> {
    pure fn index(index: Index) -> Result;
}

