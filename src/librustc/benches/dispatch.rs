use test::Bencher;

// Static/dynamic method dispatch

struct Struct {
    field: isize
}

trait Trait {
    fn method(&self) -> isize;
}

impl Trait for Struct {
    fn method(&self) -> isize {
        self.field
    }
}

#[bench]
fn trait_vtable_method_call(b: &mut Bencher) {
    let s = Struct { field: 10 };
    let t = &s as &Trait;
    b.iter(|| {
        t.method()
    });
}

#[bench]
fn trait_static_method_call(b: &mut Bencher) {
    let s = Struct { field: 10 };
    b.iter(|| {
        s.method()
    });
}
