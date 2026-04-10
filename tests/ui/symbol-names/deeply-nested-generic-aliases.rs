//@ build-pass
//@ compile-flags: -C symbol-mangling-version=v0

struct D;

type O0<T> = Option<T>;
type O1<T> = O0<T>;
type O2<T> = O1<O1<T>>;
type O3<T> = O2<O2<T>>;
type O4<T> = O3<O3<T>>;
type O5<T> = O4<O4<T>>;
type O6<T> = O5<O5<T>>;
type O7<T> = O6<O6<T>>;
type O8<T> = O7<O7<T>>;
type Q510<T> = O8<O7<O6<O5<O4<O3<O2<T>>>>>>>;

fn f<T>() {}
fn describe<T>() {
    f::<Q510<T>>()
}

fn main() {
    describe::<D>();
}
