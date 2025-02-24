use std::fmt;

fn main() {
    basic();
    diamond();
    struct_();
    replace_vptr();
    vtable_nop_cast();
    drop_principal();
    modulo_binder();
    modulo_assoc();
    bidirectional_subtyping();
}

fn vtable_nop_cast() {
    let ptr: &dyn fmt::Debug = &0;
    // We transmute things around, but the principal trait does not change, so this is allowed.
    let ptr: *const (dyn fmt::Debug + Send + Sync) = unsafe { std::mem::transmute(ptr) };
    // This cast is a NOP and should be allowed.
    let _ptr2 = ptr as *const dyn fmt::Debug;
}

fn basic() {
    trait Foo: PartialEq<i32> + fmt::Debug + Send + Sync {
        fn a(&self) -> i32 {
            10
        }

        fn z(&self) -> i32 {
            11
        }

        fn y(&self) -> i32 {
            12
        }
    }

    trait Bar: Foo {
        fn b(&self) -> i32 {
            20
        }

        fn w(&self) -> i32 {
            21
        }
    }

    trait Baz: Bar {
        fn c(&self) -> i32 {
            30
        }
    }

    impl Foo for i32 {
        fn a(&self) -> i32 {
            100
        }
    }

    impl Bar for i32 {
        fn b(&self) -> i32 {
            200
        }
    }

    impl Baz for i32 {
        fn c(&self) -> i32 {
            300
        }
    }

    let baz: &dyn Baz = &1;
    let _up: &dyn fmt::Debug = baz;
    assert_eq!(*baz, 1);
    assert_eq!(baz.a(), 100);
    assert_eq!(baz.b(), 200);
    assert_eq!(baz.c(), 300);
    assert_eq!(baz.z(), 11);
    assert_eq!(baz.y(), 12);
    assert_eq!(baz.w(), 21);

    let bar: &dyn Bar = baz;
    let _up: &dyn fmt::Debug = bar;
    assert_eq!(*bar, 1);
    assert_eq!(bar.a(), 100);
    assert_eq!(bar.b(), 200);
    assert_eq!(bar.z(), 11);
    assert_eq!(bar.y(), 12);
    assert_eq!(bar.w(), 21);

    let foo: &dyn Foo = baz;
    let _up: &dyn fmt::Debug = foo;
    assert_eq!(*foo, 1);
    assert_eq!(foo.a(), 100);
    assert_eq!(foo.z(), 11);
    assert_eq!(foo.y(), 12);

    let foo: &dyn Foo = bar;
    let _up: &dyn fmt::Debug = foo;
    assert_eq!(*foo, 1);
    assert_eq!(foo.a(), 100);
    assert_eq!(foo.z(), 11);
    assert_eq!(foo.y(), 12);
}

fn diamond() {
    trait Foo: PartialEq<i32> + fmt::Debug + Send + Sync {
        fn a(&self) -> i32 {
            10
        }

        fn z(&self) -> i32 {
            11
        }

        fn y(&self) -> i32 {
            12
        }
    }

    trait Bar1: Foo {
        fn b(&self) -> i32 {
            20
        }

        fn w(&self) -> i32 {
            21
        }
    }

    trait Bar2: Foo {
        fn c(&self) -> i32 {
            30
        }

        fn v(&self) -> i32 {
            31
        }
    }

    trait Baz: Bar1 + Bar2 {
        fn d(&self) -> i32 {
            40
        }
    }

    impl Foo for i32 {
        fn a(&self) -> i32 {
            100
        }
    }

    impl Bar1 for i32 {
        fn b(&self) -> i32 {
            200
        }
    }

    impl Bar2 for i32 {
        fn c(&self) -> i32 {
            300
        }
    }

    impl Baz for i32 {
        fn d(&self) -> i32 {
            400
        }
    }

    let baz: &dyn Baz = &1;
    let _up: &dyn fmt::Debug = baz;
    assert_eq!(*baz, 1);
    assert_eq!(baz.a(), 100);
    assert_eq!(baz.b(), 200);
    assert_eq!(baz.c(), 300);
    assert_eq!(baz.d(), 400);
    assert_eq!(baz.z(), 11);
    assert_eq!(baz.y(), 12);
    assert_eq!(baz.w(), 21);
    assert_eq!(baz.v(), 31);

    let bar1: &dyn Bar1 = baz;
    let _up: &dyn fmt::Debug = bar1;
    assert_eq!(*bar1, 1);
    assert_eq!(bar1.a(), 100);
    assert_eq!(bar1.b(), 200);
    assert_eq!(bar1.z(), 11);
    assert_eq!(bar1.y(), 12);
    assert_eq!(bar1.w(), 21);

    let bar2: &dyn Bar2 = baz;
    let _up: &dyn fmt::Debug = bar2;
    assert_eq!(*bar2, 1);
    assert_eq!(bar2.a(), 100);
    assert_eq!(bar2.c(), 300);
    assert_eq!(bar2.z(), 11);
    assert_eq!(bar2.y(), 12);
    assert_eq!(bar2.v(), 31);

    let foo: &dyn Foo = baz;
    let _up: &dyn fmt::Debug = foo;
    assert_eq!(*foo, 1);
    assert_eq!(foo.a(), 100);

    let foo: &dyn Foo = bar1;
    let _up: &dyn fmt::Debug = foo;
    assert_eq!(*foo, 1);
    assert_eq!(foo.a(), 100);

    let foo: &dyn Foo = bar2;
    let _up: &dyn fmt::Debug = foo;
    assert_eq!(*foo, 1);
    assert_eq!(foo.a(), 100);
}

fn struct_() {
    use std::rc::Rc;
    use std::sync::Arc;

    trait Foo: PartialEq<i32> + fmt::Debug + Send + Sync {
        fn a(&self) -> i32 {
            10
        }

        fn z(&self) -> i32 {
            11
        }

        fn y(&self) -> i32 {
            12
        }
    }

    trait Bar: Foo {
        fn b(&self) -> i32 {
            20
        }

        fn w(&self) -> i32 {
            21
        }
    }

    trait Baz: Bar {
        fn c(&self) -> i32 {
            30
        }
    }

    impl Foo for i32 {
        fn a(&self) -> i32 {
            100
        }
    }

    impl Bar for i32 {
        fn b(&self) -> i32 {
            200
        }
    }

    impl Baz for i32 {
        fn c(&self) -> i32 {
            300
        }
    }

    fn test_box() {
        let v = Box::new(1);

        let baz: Box<dyn Baz> = v.clone();
        assert_eq!(*baz, 1);
        assert_eq!(baz.a(), 100);
        assert_eq!(baz.b(), 200);
        assert_eq!(baz.c(), 300);
        assert_eq!(baz.z(), 11);
        assert_eq!(baz.y(), 12);
        assert_eq!(baz.w(), 21);

        let baz: Box<dyn Baz> = v.clone();
        let bar: Box<dyn Bar> = baz;
        assert_eq!(*bar, 1);
        assert_eq!(bar.a(), 100);
        assert_eq!(bar.b(), 200);
        assert_eq!(bar.z(), 11);
        assert_eq!(bar.y(), 12);
        assert_eq!(bar.w(), 21);

        let baz: Box<dyn Baz> = v.clone();
        let foo: Box<dyn Foo> = baz;
        assert_eq!(*foo, 1);
        assert_eq!(foo.a(), 100);
        assert_eq!(foo.z(), 11);
        assert_eq!(foo.y(), 12);

        let baz: Box<dyn Baz> = v.clone();
        let bar: Box<dyn Bar> = baz;
        let foo: Box<dyn Foo> = bar;
        assert_eq!(*foo, 1);
        assert_eq!(foo.a(), 100);
        assert_eq!(foo.z(), 11);
        assert_eq!(foo.y(), 12);
    }

    fn test_rc() {
        let v = Rc::new(1);

        let baz: Rc<dyn Baz> = v.clone();
        assert_eq!(*baz, 1);
        assert_eq!(baz.a(), 100);
        assert_eq!(baz.b(), 200);
        assert_eq!(baz.c(), 300);
        assert_eq!(baz.z(), 11);
        assert_eq!(baz.y(), 12);
        assert_eq!(baz.w(), 21);

        let baz: Rc<dyn Baz> = v.clone();
        let bar: Rc<dyn Bar> = baz;
        assert_eq!(*bar, 1);
        assert_eq!(bar.a(), 100);
        assert_eq!(bar.b(), 200);
        assert_eq!(bar.z(), 11);
        assert_eq!(bar.y(), 12);
        assert_eq!(bar.w(), 21);

        let baz: Rc<dyn Baz> = v.clone();
        let foo: Rc<dyn Foo> = baz;
        assert_eq!(*foo, 1);
        assert_eq!(foo.a(), 100);
        assert_eq!(foo.z(), 11);
        assert_eq!(foo.y(), 12);

        let baz: Rc<dyn Baz> = v.clone();
        let bar: Rc<dyn Bar> = baz;
        let foo: Rc<dyn Foo> = bar;
        assert_eq!(*foo, 1);
        assert_eq!(foo.a(), 100);
        assert_eq!(foo.z(), 11);
        assert_eq!(foo.y(), 12);
        assert_eq!(foo.z(), 11);
        assert_eq!(foo.y(), 12);
    }

    fn test_arc() {
        let v = Arc::new(1);

        let baz: Arc<dyn Baz> = v.clone();
        assert_eq!(*baz, 1);
        assert_eq!(baz.a(), 100);
        assert_eq!(baz.b(), 200);
        assert_eq!(baz.c(), 300);
        assert_eq!(baz.z(), 11);
        assert_eq!(baz.y(), 12);
        assert_eq!(baz.w(), 21);

        let baz: Arc<dyn Baz> = v.clone();
        let bar: Arc<dyn Bar> = baz;
        assert_eq!(*bar, 1);
        assert_eq!(bar.a(), 100);
        assert_eq!(bar.b(), 200);
        assert_eq!(bar.z(), 11);
        assert_eq!(bar.y(), 12);
        assert_eq!(bar.w(), 21);

        let baz: Arc<dyn Baz> = v.clone();
        let foo: Arc<dyn Foo> = baz;
        assert_eq!(*foo, 1);
        assert_eq!(foo.a(), 100);
        assert_eq!(foo.z(), 11);
        assert_eq!(foo.y(), 12);

        let baz: Arc<dyn Baz> = v.clone();
        let bar: Arc<dyn Bar> = baz;
        let foo: Arc<dyn Foo> = bar;
        assert_eq!(*foo, 1);
        assert_eq!(foo.a(), 100);
        assert_eq!(foo.z(), 11);
        assert_eq!(foo.y(), 12);
    }

    test_box();
    test_rc();
    test_arc();
}

fn replace_vptr() {
    trait A {
        #[allow(dead_code)]
        fn foo_a(&self);
    }

    trait B {
        #[allow(dead_code)]
        fn foo_b(&self);
    }

    trait C: A + B {
        #[allow(dead_code)]
        fn foo_c(&self);
    }

    struct S(i32);

    impl A for S {
        fn foo_a(&self) {
            unreachable!();
        }
    }

    impl B for S {
        fn foo_b(&self) {
            assert_eq!(42, self.0);
        }
    }

    impl C for S {
        fn foo_c(&self) {
            unreachable!();
        }
    }

    fn invoke_inner(b: &dyn B) {
        b.foo_b();
    }

    fn invoke_outer(c: &dyn C) {
        invoke_inner(c);
    }

    let s = S(42);
    invoke_outer(&s);
}

fn drop_principal() {
    use std::alloc::Layout;
    use std::any::Any;

    const fn yeet_principal(x: Box<dyn Any + Send>) -> Box<dyn Send> {
        x
    }

    trait Bar: Send + Sync {}

    impl<T: Send + Sync> Bar for T {}

    const fn yeet_principal_2(x: Box<dyn Bar>) -> Box<dyn Send> {
        x
    }

    struct CallMe<F: FnOnce()>(Option<F>);

    impl<F: FnOnce()> CallMe<F> {
        fn new(f: F) -> Self {
            CallMe(Some(f))
        }
    }

    impl<F: FnOnce()> Drop for CallMe<F> {
        fn drop(&mut self) {
            (self.0.take().unwrap())();
        }
    }

    fn goodbye() {
        println!("goodbye");
    }

    let x = Box::new(CallMe::new(goodbye)) as Box<dyn Any + Send>;
    let x_layout = Layout::for_value(&*x);
    let y = yeet_principal(x);
    let y_layout = Layout::for_value(&*y);
    assert_eq!(x_layout, y_layout);
    println!("before");
    drop(y);

    let x = Box::new(CallMe::new(goodbye)) as Box<dyn Bar>;
    let x_layout = Layout::for_value(&*x);
    let y = yeet_principal_2(x);
    let y_layout = Layout::for_value(&*y);
    assert_eq!(x_layout, y_layout);
    println!("before");
    drop(y);
}

// Test for <https://github.com/rust-lang/rust/issues/135316>.
fn modulo_binder() {
    trait Supertrait<T> {
        fn _print_numbers(&self, mem: &[usize; 100]) {
            println!("{mem:?}");
        }
    }
    impl<T> Supertrait<T> for () {}

    trait Trait<T, U>: Supertrait<T> + Supertrait<U> {
        fn say_hello(&self, _: &usize) {
            println!("Hello!");
        }
    }
    impl<T, U> Trait<T, U> for () {}

    (&() as &'static dyn for<'a> Trait<&'static (), &'a ()>
        as &'static dyn Trait<&'static (), &'static ()>)
        .say_hello(&0);
}

// Test for <https://github.com/rust-lang/rust/issues/135315>.
fn modulo_assoc() {
    trait Supertrait<T> {
        fn _print_numbers(&self, mem: &[usize; 100]) {
            println!("{mem:?}");
        }
    }
    impl<T> Supertrait<T> for () {}

    trait Identity {
        type Selff;
    }
    impl<Selff> Identity for Selff {
        type Selff = Selff;
    }

    trait Middle<T>: Supertrait<()> + Supertrait<T> {
        fn say_hello(&self, _: &usize) {
            println!("Hello!");
        }
    }
    impl<T> Middle<T> for () {}

    trait Trait: Middle<<() as Identity>::Selff> {}
    impl Trait for () {}

    (&() as &dyn Trait as &dyn Middle<()>).say_hello(&0);
}

fn bidirectional_subtyping() {
    // Test that transmuting between subtypes of dyn traits is fine, even in the
    // "wrong direction", i.e. going from a lower-ranked to a higher-ranked dyn trait.
    // Note that compared to the `dyn-transmute-inner-binder` test, the `for` is on the
    // *outside* here!

    trait Trait<U: ?Sized> {}
    impl<T, U: ?Sized> Trait<U> for T {}

    struct Wrapper<T: ?Sized>(T);

    let x: &dyn Trait<fn(&'static ())> = &();
    let _y: &dyn for<'a> Trait<fn(&'a ())> = unsafe { std::mem::transmute(x) };

    let x: &dyn for<'a> Trait<fn(&'a ())> = &();
    let _y: &dyn Trait<fn(&'static ())> = unsafe { std::mem::transmute(x) };

    let x: &dyn Trait<dyn Trait<fn(&'static ())>> = &();
    let _y: &dyn for<'a> Trait<dyn Trait<fn(&'a ())>> = unsafe { std::mem::transmute(x) };

    let x: &dyn for<'a> Trait<dyn Trait<fn(&'a ())>> = &();
    let _y: &dyn Trait<dyn Trait<fn(&'static ())>> = unsafe { std::mem::transmute(x) };

    // This lowers to a ptr-to-ptr cast (which behaves like a transmute)
    // and not an unsizing coercion:
    let x: *const dyn for<'a> Trait<&'a ()> = &();
    let _y: *const Wrapper<dyn Trait<&'static ()>> = x as _;
}
