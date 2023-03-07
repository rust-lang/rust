trait Trait {
    fn func1() -> Struct1<Self>; //~ ERROR E0277
    fn func2<'a>() -> Struct2<'a, Self>; //~ ERROR E0277
    fn func3() -> Struct3<Self>; //~ ERROR E0277
    fn func4() -> Struct4<Self>; //~ ERROR E0277
}

struct Struct1<T>{
    _t: std::marker::PhantomData<*const T>,
}
struct Struct2<'a, T>{
    _t: &'a T,
}
struct Struct3<T>{
    _t: T,
}

struct X<T>(T);

struct Struct4<T>{
    _t: X<T>,
}

struct Struct5<T: ?Sized>{
    _t: X<T>, //~ ERROR E0277
}

fn main() {}
