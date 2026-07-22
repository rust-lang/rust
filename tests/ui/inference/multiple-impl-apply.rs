struct Foo {
    inner: u32,
}

struct Bar {
    inner: u32,
}

#[derive(Clone, Copy)]
struct Baz {
    inner: u32,
}

impl From<Baz> for Bar {
    fn from(other: Baz) -> Self {
        Self {
            inner: other.inner,
        }
    }
}

impl From<Baz> for Foo {
    fn from(other: Baz) -> Self {
        Self {
            inner: other.inner,
        }
    }
}

fn main() {
    let x: Baz = Baz { inner: 42 };

    // DOESN'T Compile: Multiple options!
    let y = x.into(); //~ ERROR E0283

    let y_1: Foo = x.into();
    let y_2: Bar = x.into();

    let z_1 = Foo::from(y_1);
    let z_2 = Bar::from(y_2);

    // No type annotations needed, the compiler KNOWS the type must be `Foo`!
    let m = magic_foo(x);
}

fn magic_foo(arg: Baz) -> Foo {
    arg.into()
}

struct Value;

impl PartialEq<Value> for u32 {
    fn eq(&self, _: &Value) -> bool {
        false
    }
}

impl PartialEq<u32> for Value {
    fn eq(&self, _: &u32) -> bool {
        false
    }
}

// https://github.com/rust-lang/rust/issues/156004
fn partial_eq_with_infer_cast_on_rhs() {
    let n: u32 = 17;
    let _ = n == 42usize as _; //~ ERROR E0283
}

fn partial_eq_with_infer_cast_on_lhs() {
    let n: u32 = 17;
    let _ = 42usize as _ == n; //~ ERROR E0283
}
