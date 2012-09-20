// Core operators and kinds.

#[lang="const"]
trait Const {
    // Empty.
}

#[lang="copy"]
trait Copy {
    // Empty.
}

#[lang="send"]
trait Send {
    // Empty.
}

#[lang="owned"]
trait Owned {
    // Empty.
}

#[cfg(stage0)]
#[lang="add"]
trait Add<RHS,Result> {
    pure fn add(rhs: RHS) -> Result;
}

#[cfg(stage1)]
#[cfg(stage2)]
#[lang="add"]
trait Add<RHS,Result> {
    pure fn add(rhs: &RHS) -> Result;
}

#[cfg(stage0)]
#[lang="sub"]
trait Sub<RHS,Result> {
    pure fn sub(rhs: RHS) -> Result;
}

#[cfg(stage1)]
#[cfg(stage2)]
#[lang="sub"]
trait Sub<RHS,Result> {
    pure fn sub(rhs: &RHS) -> Result;
}

#[cfg(stage0)]
#[lang="mul"]
trait Mul<RHS,Result> {
    pure fn mul(rhs: RHS) -> Result;
}

#[cfg(stage1)]
#[cfg(stage2)]
#[lang="mul"]
trait Mul<RHS,Result> {
    pure fn mul(rhs: &RHS) -> Result;
}

#[cfg(stage0)]
#[lang="div"]
trait Div<RHS,Result> {
    pure fn div(rhs: RHS) -> Result;
}

#[cfg(stage1)]
#[cfg(stage2)]
#[lang="div"]
trait Div<RHS,Result> {
    pure fn div(rhs: &RHS) -> Result;
}

#[cfg(stage0)]
#[lang="modulo"]
trait Modulo<RHS,Result> {
    pure fn modulo(rhs: RHS) -> Result;
}

#[cfg(stage1)]
#[cfg(stage2)]
#[lang="modulo"]
trait Modulo<RHS,Result> {
    pure fn modulo(rhs: &RHS) -> Result;
}

#[lang="neg"]
trait Neg<Result> {
    pure fn neg() -> Result;
}

#[cfg(stage0)]
#[lang="bitand"]
trait BitAnd<RHS,Result> {
    pure fn bitand(rhs: RHS) -> Result;
}

#[cfg(stage1)]
#[cfg(stage2)]
#[lang="bitand"]
trait BitAnd<RHS,Result> {
    pure fn bitand(rhs: &RHS) -> Result;
}

#[cfg(stage0)]
#[lang="bitor"]
trait BitOr<RHS,Result> {
    pure fn bitor(rhs: RHS) -> Result;
}

#[cfg(stage1)]
#[cfg(stage2)]
#[lang="bitor"]
trait BitOr<RHS,Result> {
    pure fn bitor(rhs: &RHS) -> Result;
}

#[cfg(stage0)]
#[lang="bitxor"]
trait BitXor<RHS,Result> {
    pure fn bitxor(rhs: RHS) -> Result;
}

#[cfg(stage1)]
#[cfg(stage2)]
#[lang="bitxor"]
trait BitXor<RHS,Result> {
    pure fn bitxor(rhs: &RHS) -> Result;
}

#[cfg(stage0)]
#[lang="shl"]
trait Shl<RHS,Result> {
    pure fn shl(rhs: RHS) -> Result;
}

#[cfg(stage1)]
#[cfg(stage2)]
#[lang="shl"]
trait Shl<RHS,Result> {
    pure fn shl(rhs: &RHS) -> Result;
}

#[cfg(stage0)]
#[lang="shr"]
trait Shr<RHS,Result> {
    pure fn shr(rhs: RHS) -> Result;
}

#[cfg(stage1)]
#[cfg(stage2)]
#[lang="shr"]
trait Shr<RHS,Result> {
    pure fn shr(rhs: &RHS) -> Result;
}

#[lang="index"]
trait Index<Index,Result> {
    pure fn index(index: Index) -> Result;
}

