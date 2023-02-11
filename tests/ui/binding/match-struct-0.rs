// run-pass

struct Foo{
    f : isize,
}

pub fn main() {
    let f = Foo{f: 1};
    match f {
        Foo{f: 0} => panic!(),
        Foo{..} => (),
    }
    match f {
        Foo{f: 0} => panic!(),
        Foo{f: _f} => (),
    }
    match f {
        Foo{f: 0} => panic!(),
        _ => (),
    }
}
