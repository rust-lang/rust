#[repr(transparent)]
enum OkButUnstableEnum { //~ ERROR transparent enums are unstable
    Foo((), String, ()),
}

fn main() {}
