// Core operators and kinds.

#[allow(non_camel_case_types)];

#[cfg(notest)]
#[lang="const"]
trait const {
    // Empty.
}

#[cfg(notest)]
#[lang="copy"]
trait copy {
    // Empty.
}

#[cfg(notest)]
#[lang="send"]
trait send {
    // Empty.
}

#[cfg(notest)]
#[lang="owned"]
trait owned {
    // Empty.
}

#[cfg(notest)]
#[lang="add"]
trait add<RHS,Result> {
    pure fn add(rhs: RHS) -> Result;
}

#[cfg(notest)]
#[lang="sub"]
trait sub<RHS,Result> {
    pure fn sub(rhs: RHS) -> Result;
}

#[cfg(notest)]
#[lang="mul"]
trait mul<RHS,Result> {
    pure fn mul(rhs: RHS) -> Result;
}

#[cfg(notest)]
#[lang="div"]
trait div<RHS,Result> {
    pure fn div(rhs: RHS) -> Result;
}

#[cfg(notest)]
#[lang="modulo"]
trait modulo<RHS,Result> {
    pure fn modulo(rhs: RHS) -> Result;
}

#[cfg(notest)]
#[lang="neg"]
trait neg<Result> {
    pure fn neg() -> Result;
}

#[cfg(notest)]
#[lang="bitand"]
trait bitand<RHS,Result> {
    pure fn bitand(rhs: RHS) -> Result;
}

#[cfg(notest)]
#[lang="bitor"]
trait bitor<RHS,Result> {
    pure fn bitor(rhs: RHS) -> Result;
}

#[cfg(notest)]
#[lang="bitxor"]
trait bitxor<RHS,Result> {
    pure fn bitxor(rhs: RHS) -> Result;
}

#[cfg(notest)]
#[lang="shl"]
trait shl<RHS,Result> {
    pure fn shl(rhs: RHS) -> Result;
}

#[cfg(notest)]
#[lang="shr"]
trait shr<RHS,Result> {
    pure fn shr(rhs: RHS) -> Result;
}

#[cfg(notest)]
#[lang="index"]
trait index<Index,Result> {
    pure fn index(index: Index) -> Result;
}

