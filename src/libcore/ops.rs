// Core operators and kinds.

#[cfg(notest)]
#[lang="const"]
trait Const {
    // Empty.
}

#[cfg(notest)]
#[lang="copy"]
trait Copy {
    // Empty.
}

#[cfg(notest)]
#[lang="send"]
trait Send {
    // Empty.
}

#[cfg(notest)]
#[lang="owned"]
trait Owned {
    // Empty.
}

#[cfg(notest)]
#[lang="add"]
trait Add<RHS,Result> {
    pure fn add(rhs: RHS) -> Result;
}

#[cfg(notest)]
#[lang="sub"]
trait Sub<RHS,Result> {
    pure fn sub(rhs: RHS) -> Result;
}

#[cfg(notest)]
#[lang="mul"]
trait Mul<RHS,Result> {
    pure fn mul(rhs: RHS) -> Result;
}

#[cfg(notest)]
#[lang="div"]
trait Div<RHS,Result> {
    pure fn div(rhs: RHS) -> Result;
}

#[cfg(notest)]
#[lang="modulo"]
trait Modulo<RHS,Result> {
    pure fn modulo(rhs: RHS) -> Result;
}

#[cfg(notest)]
#[lang="neg"]
trait Neg<Result> {
    pure fn neg() -> Result;
}

#[cfg(notest)]
#[lang="bitand"]
trait BitAnd<RHS,Result> {
    pure fn bitand(rhs: RHS) -> Result;
}

#[cfg(notest)]
#[lang="bitor"]
trait BitOr<RHS,Result> {
    pure fn bitor(rhs: RHS) -> Result;
}

#[cfg(notest)]
#[lang="bitxor"]
trait BitXor<RHS,Result> {
    pure fn bitxor(rhs: RHS) -> Result;
}

#[cfg(notest)]
#[lang="shl"]
trait Shl<RHS,Result> {
    pure fn shl(rhs: RHS) -> Result;
}

#[cfg(notest)]
#[lang="shr"]
trait Shr<RHS,Result> {
    pure fn shr(rhs: RHS) -> Result;
}

#[cfg(notest)]
#[lang="index"]
trait Index<Index,Result> {
    pure fn index(index: Index) -> Result;
}

