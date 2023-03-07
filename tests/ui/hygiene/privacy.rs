#![feature(decl_macro)]

mod foo {
    fn f() {}

    pub macro m($e:expr) {
        f();
        self::f();
        ::foo::f();
        $e
    }
}

fn main() {
    foo::m!(
        foo::f() //~ ERROR `f` is private
    );
}
