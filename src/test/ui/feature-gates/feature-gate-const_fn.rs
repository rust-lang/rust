// Test use of const fn without feature gate.

const fn foo() -> usize { 0 } //~ ERROR const fn is unstable

trait Foo {
    const fn foo() -> u32; //~ ERROR const fn is unstable
                           //~| ERROR trait fns cannot be declared const
    const fn bar() -> u32 { 0 } //~ ERROR const fn is unstable
                                //~| ERROR trait fns cannot be declared const
}

impl Foo {
    const fn baz() -> u32 { 0 } //~ ERROR const fn is unstable
}

impl Foo for u32 {
    const fn foo() -> u32 { 0 } //~ ERROR const fn is unstable
                                //~| ERROR trait fns cannot be declared const
}

static FOO: usize = foo();
const BAR: usize = foo();

macro_rules! constant {
    ($n:ident: $t:ty = $v:expr) => {
        const $n: $t = $v;
    }
}

constant! {
    BAZ: usize = foo()
}

fn main() {
    let x: [usize; foo()] = [];
}
