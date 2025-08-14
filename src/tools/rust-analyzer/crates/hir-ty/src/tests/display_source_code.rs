use super::check_types_source_code;

#[test]
fn qualify_path_to_submodule() {
    check_types_source_code(
        r#"
mod foo {
    pub struct Foo;
}

fn bar() {
    let foo: foo::Foo = foo::Foo;
    foo;
} //^^^ foo::Foo

"#,
    );
}

#[test]
fn omit_default_type_parameters() {
    check_types_source_code(
        r#"
struct Foo<T = u8> { t: T }
fn main() {
    let foo = Foo { t: 5u8 };
    foo;
} //^^^ Foo
"#,
    );

    check_types_source_code(
        r#"
struct Foo<K, T = u8> { k: K, t: T }
fn main() {
    let foo = Foo { k: 400, t: 5u8 };
    foo;
} //^^^ Foo<i32>
"#,
    );
}

#[test]
fn render_raw_ptr_impl_ty() {
    check_types_source_code(
        r#"
//- minicore: sized
trait Unpin {}
fn foo() -> *const (impl Unpin + Sized) { loop {} }
fn main() {
    let foo = foo();
    foo;
} //^^^ *const impl Unpin
"#,
    );
}

#[test]
fn render_dyn_ty_independent_of_order() {
    check_types_source_code(
        r#"
auto trait Send {}
trait A {
    type Assoc;
}
trait B: A {}

fn test<'a>(
    _: &(dyn A<Assoc = ()> + Send),
  //^ &(dyn A<Assoc = ()> + Send)
    _: &'a (dyn Send + A<Assoc = ()>),
  //^ &'a (dyn A<Assoc = ()> + Send)
    _: &dyn B<Assoc = ()>,
  //^ &(dyn B<Assoc = ()>)
) {}
        "#,
    );
}

#[test]
fn render_dyn_for_ty() {
    // FIXME
    check_types_source_code(
        r#"
trait Foo<'a> {}

fn foo(foo: &dyn for<'a> Foo<'a>) {}
    // ^^^ &dyn Foo<'?>
"#,
    );
}

#[test]
fn sized_bounds_apit() {
    check_types_source_code(
        r#"
//- minicore: sized
trait Foo {}
trait Bar<T> {}
struct S<T>;
fn test(
    a: impl Foo,
    b: impl Foo + Sized,
    c: &(impl Foo + ?Sized),
    d: S<impl Foo>,
    ref_any: &impl ?Sized,
    empty: impl,
) {
    a;
  //^ impl Foo
    b;
  //^ impl Foo
    c;
  //^ &impl Foo + ?Sized
    d;
  //^ S<impl Foo>
    ref_any;
  //^^^^^^^ &impl ?Sized
    empty;
} //^^^^^ impl Sized
"#,
    );
}

#[test]
fn sized_bounds_rpit() {
    check_types_source_code(
        r#"
//- minicore: sized
trait Foo {}
fn foo1() -> impl Foo { loop {} }
fn foo2() -> impl Foo + Sized { loop {} }
fn foo3() -> impl Foo + ?Sized { loop {} }
fn test() {
    let foo = foo1();
    foo;
  //^^^ impl Foo
    let foo = foo2();
    foo;
  //^^^ impl Foo
    let foo = foo3();
    foo;
} //^^^ impl Foo + ?Sized
"#,
    );
}

#[test]
fn parenthesize_ptr_rpit_sized_bounds() {
    check_types_source_code(
        r#"
//- minicore: sized
trait Foo {}
fn foo1() -> *const impl Foo { loop {} }
fn foo2() -> *const (impl Foo + Sized) { loop {} }
fn foo3() -> *const (impl Sized + Foo) { loop {} }
fn foo4() -> *const (impl Foo + ?Sized) { loop {} }
fn foo5() -> *const (impl ?Sized + Foo) { loop {} }
fn test() {
    let foo = foo1();
    foo;
  //^^^ *const impl Foo
    let foo = foo2();
    foo;
  //^^^ *const impl Foo
    let foo = foo3();
    foo;
  //^^^ *const impl Foo
    let foo = foo4();
    foo;
  //^^^ *const (impl Foo + ?Sized)
    let foo = foo5();
    foo;
} //^^^ *const (impl Foo + ?Sized)
"#,
    );
}

#[test]
fn sized_bounds_impl_traits_in_fn_signature() {
    check_types_source_code(
        r#"
//- minicore: sized
trait Foo {}
fn test(
    a: fn(impl Foo) -> impl Foo,
    b: fn(impl Foo + Sized) -> impl Foo + Sized,
    c: fn(&(impl Foo + ?Sized)) -> &(impl Foo + ?Sized),
) {
    a;
  //^ fn(impl Foo) -> impl Foo
    b;
  //^ fn(impl Foo) -> impl Foo
    c;
} //^ fn(&impl Foo + ?Sized) -> &impl Foo + ?Sized
"#,
    );
}

#[test]
fn projection_type_correct_arguments_order() {
    check_types_source_code(
        r#"
trait Foo<T> {
    type Assoc<U>;
}
fn f<T: Foo<i32>>(a: T::Assoc<usize>) {
    a;
  //^ <T as Foo<i32>>::Assoc<usize>
}
"#,
    );
}

#[test]
fn generic_associated_type_binding_in_impl_trait() {
    check_types_source_code(
        r#"
//- minicore: sized
trait Foo<T> {
    type Assoc<U>;
}
fn f(a: impl Foo<i8, Assoc<i16> = i32>) {
    a;
  //^ impl Foo<i8, Assoc<i16> = i32>
}
        "#,
    );
}

#[test]
fn fn_def_is_shown_as_fn_ptr() {
    check_types_source_code(
        r#"
fn foo(_: i32) -> i64 { 42 }
struct S<T>(T);
enum E { A(usize) }
fn test() {
    let f = foo;
      //^ fn(i32) -> i64
    let f = S::<i8>;
      //^ fn(i8) -> S<i8>
    let f = E::A;
      //^ fn(usize) -> E
}
"#,
    );
}
