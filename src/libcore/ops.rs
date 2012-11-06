// Core operators and kinds.

#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

#[lang="const"]
pub trait Const {
    // Empty.
}

#[lang="copy"]
pub trait Copy {
    // Empty.
}

#[lang="send"]
pub trait Send {
    // Empty.
}

#[lang="owned"]
pub trait Owned {
    // Empty.
}

#[lang="drop"]
pub trait Drop {
    fn finalize();  // XXX: Rename to "drop"? --pcwalton
}

#[lang="add"]
pub trait Add<RHS,Result> {
    pure fn add(rhs: &RHS) -> Result;
}

#[lang="sub"]
pub trait Sub<RHS,Result> {
    pure fn sub(rhs: &RHS) -> Result;
}

#[lang="mul"]
pub trait Mul<RHS,Result> {
    pure fn mul(rhs: &RHS) -> Result;
}

#[lang="div"]
pub trait Div<RHS,Result> {
    pure fn div(rhs: &RHS) -> Result;
}

#[lang="modulo"]
pub trait Modulo<RHS,Result> {
    pure fn modulo(rhs: &RHS) -> Result;
}

#[lang="neg"]
pub trait Neg<Result> {
    pure fn neg() -> Result;
}

#[lang="bitand"]
pub trait BitAnd<RHS,Result> {
    pure fn bitand(rhs: &RHS) -> Result;
}

#[lang="bitor"]
pub trait BitOr<RHS,Result> {
    pure fn bitor(rhs: &RHS) -> Result;
}

#[lang="bitxor"]
pub trait BitXor<RHS,Result> {
    pure fn bitxor(rhs: &RHS) -> Result;
}

#[lang="shl"]
pub trait Shl<RHS,Result> {
    pure fn shl(rhs: &RHS) -> Result;
}

#[lang="shr"]
pub trait Shr<RHS,Result> {
    pure fn shr(rhs: &RHS) -> Result;
}

#[lang="index"]
pub trait Index<Index,Result> {
    pure fn index(index: Index) -> Result;
}

