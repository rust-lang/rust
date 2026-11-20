// Regression test for #46314

#![feature(decl_macro)]

struct NonCopy(String);

struct Wrapper {
    inner: NonCopy,
}

macro inner_copy($wrapper:ident) {
    $wrapper.inner
}

fn main() {
    let wrapper = Wrapper {
        inner: NonCopy("foo".into()),
    };
    assert_two_non_copy(
        inner_copy!(wrapper),
        wrapper.inner,
        //~^ ERROR use of moved value: `wrapper.inner` [E0382]
    );
}

fn assert_two_non_copy(a: NonCopy, b: NonCopy) {
    assert_eq!(a.0, b.0);
}
