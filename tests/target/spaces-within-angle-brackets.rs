// rustfmt-spaces_within_angle_brackets: true

struct Foo< T > {
    i: T,
}

struct Bar< T, E > {
    i: T,
    e: E,
}

struct Foo< 'a > {
    i: &'a str,
}

enum E< T > {
    T(T),
}

enum E< T, S > {
    T(T),
    S(S),
}

fn foo< T >(a: T) {
    foo::< u32 >(10);
}

fn foo< T, E >(a: T, b: E) {
    foo::< u32, str >(10, "bar");
}

fn foo< T: Send, E: Send >(a: T, b: E) {

    foo::< u32, str >(10, "bar");

    let opt: Option< u32 >;
    let res: Result< u32, String >;
}

fn foo< 'a >(a: &'a str) {
    foo("foo");
}

fn foo< 'a, 'b >(a: &'a str, b: &'b str) {
    foo("foo", "bar");
}

impl Foo {
    fn bar() {
        < Foo as Foo >::bar();
    }
}

trait MyTrait< A, D > {}
impl< A: Send, D: Send > MyTrait< A, D > for Foo {}

fn foo()
where
    for< 'a > u32: 'a,
{
}
