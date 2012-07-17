fn copy1<T: copy>(t: T) -> fn@() -> T {
    fn@() -> T { t } //~ ERROR not an owned value
}

fn copy2<T: copy owned>(t: T) -> fn@() -> T {
    fn@() -> T { t }
}

fn main() {
    let x = &3;
    copy2(&x); //~ ERROR instantiating a type parameter with an incompatible type

    copy2(@3);
    copy2(@&x); //~ ERROR instantiating a type parameter with an incompatible type

    copy2(fn@() {});
    copy2(fn~() {}); //~ WARNING instantiating copy type parameter with a not implicitly copyable type
    copy2(fn&() {}); //~ ERROR instantiating a type parameter with an incompatible type
}
