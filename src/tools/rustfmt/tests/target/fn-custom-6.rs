// rustfmt-brace_style: PreferSameLine
// Test different indents.

fn foo(a: Aaaaaaaaaaaaaa, b: Bbbbbbbbbbbbbb) {
    foo();
}

fn bar(
    a: Aaaaaaaaaaaaaa,
    b: Bbbbbbbbbbbbbb,
    c: Cccccccccccccccccc,
    d: Dddddddddddddddd,
    e: Eeeeeeeeeeeeeee,
) {
    bar();
}

fn foo(a: Aaaaaaaaaaaaaa, b: Bbbbbbbbbbbbbb) -> String {
    foo();
}

fn bar(
    a: Aaaaaaaaaaaaaa,
    b: Bbbbbbbbbbbbbb,
    c: Cccccccccccccccccc,
    d: Dddddddddddddddd,
    e: Eeeeeeeeeeeeeee,
) -> String {
    bar();
}

fn foo(a: Aaaaaaaaaaaaaa, b: Bbbbbbbbbbbbbb)
where
    T: UUUUUUUUUUU, {
    foo();
}

fn bar(
    a: Aaaaaaaaaaaaaa,
    b: Bbbbbbbbbbbbbb,
    c: Cccccccccccccccccc,
    d: Dddddddddddddddd,
    e: Eeeeeeeeeeeeeee,
) where
    T: UUUUUUUUUUU, {
    bar();
}

fn foo(a: Aaaaaaaaaaaaaa, b: Bbbbbbbbbbbbbb) -> String
where
    T: UUUUUUUUUUU, {
    foo();
}

fn bar(
    a: Aaaaaaaaaaaaaa,
    b: Bbbbbbbbbbbbbb,
    c: Cccccccccccccccccc,
    d: Dddddddddddddddd,
    e: Eeeeeeeeeeeeeee,
) -> String
where
    T: UUUUUUUUUUU, {
    bar();
}

trait Test {
    fn foo(a: u8) {}

    fn bar(a: u8) -> String {}
}
