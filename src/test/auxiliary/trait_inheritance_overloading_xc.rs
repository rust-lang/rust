pub trait MyNum : Add<self,self>, Sub<self,self>, Mul<self,self> {
}

pub impl int : MyNum {
    pure fn add(other: &int) -> int { self + *other }
    pure fn sub(other: &int) -> int { self - *other }
    pure fn mul(other: &int) -> int { self * *other }
}

