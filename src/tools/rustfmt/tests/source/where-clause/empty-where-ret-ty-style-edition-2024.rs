//@ rustfmt-style_edition: 2024

fn implicit_ret_ty_1() where {}
fn implicit_ret_ty_2() /* comment */ where {}
fn implicit_ret_ty_3() where /* comment */ {}
fn implicit_ret_ty_4() /* comment */ where /* comment */ {}

fn foo() -> () where {}
fn bar() -> () /* comment */ where {}
fn baz() -> () where /* comment */ {}
fn qux() -> () /* comment */ where /* comment */ {}

fn implicit_line_before() // c1
where {}
fn implicit_line_after() where // c2
{}
fn implicit_line_both() // c1
where // c2
{}

fn explicit_line_before() -> () // c1
where {}
fn explicit_line_after() -> () where // c2
{}
fn line_comments() -> () // c1
where // c2
{}

foo! {
    fn implicit_ret_ty() where {}
}

bar! {
    fn explicit_ret_ty() -> () where {}
}

i1!(fn implicit_ret_ty_1() where {});
i2!(fn implicit_ret_ty_2() /* comment */ where {});
i3!(fn implicit_ret_ty_3() where /* comment */ {});
i4!(fn implicit_ret_ty_4() /* comment */ where /* comment */ {});

i1!(fn explicit_ret_ty_1() -> i32 where {});
i2!(fn explicit_ret_ty_2() -> i32 /* comment */ where {});
i3!(fn explicit_ret_ty_3() -> i32 where /* comment */ {});
i4!(fn explicit_ret_ty_4() -> i32 /* comment */ where /* comment */ {});

extern "C" {
    fn implicit_ret_ty_1() where;
    fn implicit_ret_ty_2() /* comment */ where;
    fn implicit_ret_ty_3() where /* comment */;
    fn implicit_ret_ty_4() /* comment */ where /* comment */;

    fn explicit_ret_ty_1() -> i32 where;
    fn explicit_ret_ty_2() -> i32 /* comment */ where;
    fn explicit_ret_ty_3() -> i32 where /* comment */;
    fn explicit_ret_ty_4() -> i32 /* comment */ where /* comment */;
}

trait MyTrait {
    fn implicit_ret_ty_1() where;
    fn implicit_ret_ty_2() /* comment */ where;
    fn implicit_ret_ty_3() where /* comment */;
    fn implicit_ret_ty_4() /* comment */ where /* comment */;

    fn explicit_ret_ty_1() -> i32 where;
    fn explicit_ret_ty_2() -> i32 /* comment */ where;
    fn explicit_ret_ty_3() -> i32 where /* comment */;
    fn explicit_ret_ty_4() -> i32 /* comment */ where /* comment */;
}

macro_rules! m1 {
    () => {
        fn implicit_ret_ty_1() where;
        fn implicit_ret_ty_2() /* comment */ where;
        fn implicit_ret_ty_3() where /* comment */;
        fn implicit_ret_ty_4() /* comment */ where /* comment */;

        fn explicit_ret_ty_1() -> i32 where;
        fn explicit_ret_ty_2() -> i32 /* comment */ where;
        fn explicit_ret_ty_3() -> i32 where /* comment */;
        fn explicit_ret_ty_4() -> i32 /* comment */ where /* comment */;
    };
}
