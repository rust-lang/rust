// rustfmt-fn_args_layout: Block
// rustfmt-fn_arg_one_line: true

fn foo() {
    foo();
}

fn foo(a: Aaaaaaaaaaaaaa, b: Bbbbbbbbbbbbbb) {
    foo();
}

fn bar(a: Aaaaaaaaaaaaaa, b: Bbbbbbbbbbbbbb, c: Cccccccccccccccccc, d: Dddddddddddddddd, e: Eeeeeeeeeeeeeee) {
    bar();
}

fn foo(a: Aaaaaaaaaaaaaa, b: Bbbbbbbbbbbbbb) -> String {
    foo();
}

fn bar(a: Aaaaaaaaaaaaaa, b: Bbbbbbbbbbbbbb, c: Cccccccccccccccccc, d: Dddddddddddddddd, e: Eeeeeeeeeeeeeee) -> String {
    bar();
}

fn foo(a: u8 /* Comment 1 */, b: u8 /* Comment 2 */) -> u8 {
    bar()
}

fn foo(a: u8 /* Comment 1 */, b: Bbbbbbbbbbbbbb, c: Cccccccccccccccccc, d: Dddddddddddddddd, e: Eeeeeeeeeeeeeee /* Comment 2 */) -> u8 {
    bar()
}

fn bar(a: Aaaaaaaaaaaaaa, b: Bbbbbbbbbbbbbb, c: Cccccccccccccccccc, d: Dddddddddddddddd, e: Eeeeeeeeeeeeeee) -> String where X: Fooooo, Y: Baaar  {
    bar();
}

fn foo() -> T {
    foo();
}

fn foo() -> T where X: Foooo, Y: Baaar {
    foo();
}

trait Test {
    fn foo(a: u8) {}

    fn bar(a: Aaaaaaaaaaaaaa, b: Bbbbbbbbbbbbbb, c: Cccccccccccccccccc, d: Dddddddddddddddd, e: Eeeeeeeeeeeeeee) -> String {}
}
