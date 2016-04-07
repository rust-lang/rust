// rustfmt-fn_args_layout: BlockAlways

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

trait Test {
    fn foo(a: u8) {}

    fn bar(a: u8) -> String {}
}
