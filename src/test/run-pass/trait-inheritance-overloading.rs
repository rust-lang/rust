trait MyNum : Add<self,self>, Sub<self,self>, Mul<self,self> {
}

impl int : MyNum {
    pure fn add(other: &int) -> int { self + *other }
    pure fn sub(other: &int) -> int { self - *other }
    pure fn mul(other: &int) -> int { self * *other }
}

fn f<T:Copy MyNum>(x: T, y: T) -> (T, T, T) {
    return (x + y, x - y, x * y);
}

fn main() {
    let (x, y) = (3, 5);
    let (a, b, c) = f(x, y);
    assert a == 8;
    assert b == -2;
    assert c == 15;
}

