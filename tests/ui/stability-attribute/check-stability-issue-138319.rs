//@ check-pass
fn _foo() {
    _Bar { //~ WARNING use of deprecated struct `_Bar`: reason
        #[expect(deprecated)]
        foo: 0,
    };
}

#[deprecated = "reason"]
struct _Bar {
    foo: u32,
}

fn _foo2() {
    #[expect(deprecated)]
    _Bar2 {
        foo2: 0,
    };
}

#[deprecated = "reason"]
struct _Bar2 {
    foo2: u32,
}

fn _foo3() {
    _Bar3 {
        #[expect(deprecated)]
        foo3: 0,
    };
}

struct _Bar3 {
    #[deprecated = "reason"]
    foo3: u32,
}


fn main() {}
