// revisions: opt debug
//[opt] compile-flags: -O

//[opt] build-pass
//[debug] build-fail

struct Foo<T>(T);

impl<T> Foo<T> {
    const BAR: () = if std::mem::size_of::<T>() == 0 {
        panic!()
        //[debug]~^ ERROR evaluation of `Foo::<()>::BAR` failed
    };
}

#[inline(never)]
fn bop<T>() {
    Foo::<T>::BAR;
}

fn fop<T>() {
    if false {
        bop::<T>();
    }
}

fn main() {
    fop::<u32>();
    fop::<()>();
}
