#![feature(trait_upcasting)]
#![allow(incomplete_features)]

fn main() {
    basic();
    diamond();
    struct_();
    replace_vptr();
    vtable_mismatch_nop_cast();
}

fn vtable_mismatch_nop_cast() {
    let ptr: &dyn std::fmt::Display = &0;
    // Even though the vtable is for the wrong trait, this cast doesn't actually change the needed
    // vtable so it should still be allowed.
    let ptr: *const (dyn std::fmt::Debug + Send + Sync) = unsafe { std::mem::transmute(ptr) };
    let _ptr2 = ptr as *const dyn std::fmt::Debug;
}

fn basic() {
    trait Foo: PartialEq<i32> + std::fmt::Debug + Send + Sync {
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
    let _: &dyn std::fmt::Debug = baz;
    assert_eq!(*baz, 1);
    assert_eq!(baz.a(), 100);
    assert_eq!(baz.b(), 200);
    assert_eq!(baz.c(), 300);
    assert_eq!(baz.z(), 11);
    assert_eq!(baz.y(), 12);
    assert_eq!(baz.w(), 21);

    let bar: &dyn Bar = baz;
    let _: &dyn std::fmt::Debug = bar;
    assert_eq!(*bar, 1);
    assert_eq!(bar.a(), 100);
    assert_eq!(bar.b(), 200);
    assert_eq!(bar.z(), 11);
    assert_eq!(bar.y(), 12);
    assert_eq!(bar.w(), 21);

    let foo: &dyn Foo = baz;
    let _: &dyn std::fmt::Debug = foo;
    assert_eq!(*foo, 1);
    assert_eq!(foo.a(), 100);
    assert_eq!(foo.z(), 11);
    assert_eq!(foo.y(), 12);

    let foo: &dyn Foo = bar;
    let _: &dyn std::fmt::Debug = foo;
    assert_eq!(*foo, 1);
    assert_eq!(foo.a(), 100);
    assert_eq!(foo.z(), 11);
    assert_eq!(foo.y(), 12);
}

fn diamond() {
    trait Foo: PartialEq<i32> + std::fmt::Debug + Send + Sync {
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
    let _: &dyn std::fmt::Debug = baz;
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
    let _: &dyn std::fmt::Debug = bar1;
    assert_eq!(*bar1, 1);
    assert_eq!(bar1.a(), 100);
    assert_eq!(bar1.b(), 200);
    assert_eq!(bar1.z(), 11);
    assert_eq!(bar1.y(), 12);
    assert_eq!(bar1.w(), 21);

    let bar2: &dyn Bar2 = baz;
    let _: &dyn std::fmt::Debug = bar2;
    assert_eq!(*bar2, 1);
    assert_eq!(bar2.a(), 100);
    assert_eq!(bar2.c(), 300);
    assert_eq!(bar2.z(), 11);
    assert_eq!(bar2.y(), 12);
    assert_eq!(bar2.v(), 31);

    let foo: &dyn Foo = baz;
    let _: &dyn std::fmt::Debug = foo;
    assert_eq!(*foo, 1);
    assert_eq!(foo.a(), 100);

    let foo: &dyn Foo = bar1;
    let _: &dyn std::fmt::Debug = foo;
    assert_eq!(*foo, 1);
    assert_eq!(foo.a(), 100);

    let foo: &dyn Foo = bar2;
    let _: &dyn std::fmt::Debug = foo;
    assert_eq!(*foo, 1);
    assert_eq!(foo.a(), 100);
}

fn struct_() {
    use std::rc::Rc;
    use std::sync::Arc;

    trait Foo: PartialEq<i32> + std::fmt::Debug + Send + Sync {
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
