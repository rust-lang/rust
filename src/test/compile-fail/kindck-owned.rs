fn copy1<T: Copy>(t: T) -> fn@() -> T {
    fn@() -> T { t } //~ ERROR value may contain borrowed pointers
}

fn copy2<T: Copy Owned>(t: T) -> fn@() -> T {
    fn@() -> T { t }
}

fn main() {
    let x = &3;
    copy2(&x); //~ ERROR missing `owned`

    copy2(@3);
    copy2(@&x); //~ ERROR missing `owned`

    copy2(fn@() {});
    copy2(fn~() {}); //~ WARNING instantiating copy type parameter with a not implicitly copyable type
    copy2(fn&() {}); //~ ERROR missing `copy owned`
}
