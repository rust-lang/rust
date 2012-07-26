// Core operators and kinds.

#[lang="const"]
trait const {
    // Empty.
}

#[lang="copy"]
trait copy {
    // Empty.
}

#[lang="send"]
trait send {
    // Empty.
}

#[lang="owned"]
trait owned {
    // Empty.
}

#[lang="add"]
trait add<RHS,Result> {
    pure fn add(rhs: RHS) -> Result;
}

#[lang="sub"]
trait sub<RHS,Result> {
    pure fn sub(rhs: RHS) -> Result;
}

#[lang="mul"]
trait mul<RHS,Result> {
    pure fn mul(rhs: RHS) -> Result;
}

#[lang="div"]
trait div<RHS,Result> {
    pure fn div(rhs: RHS) -> Result;
}

#[lang="modulo"]
trait modulo<RHS,Result> {
    pure fn modulo(rhs: RHS) -> Result;
}

#[lang="neg"]
trait neg<RHS,Result> {
    pure fn neg(rhs: RHS) -> Result;
}

#[lang="bitops"]
trait bitops<RHS,BitCount,Result> {
    pure fn and(rhs: RHS) -> Result;
    pure fn or(rhs: RHS) -> Result;
    pure fn xor(rhs: RHS) -> Result;
    pure fn shl(n: BitCount) -> Result;
    pure fn shr(n: BitCount) -> Result;
}

#[lang="index"]
trait index<Index,Result> {
    pure fn index(index: Index) -> Result;
}

