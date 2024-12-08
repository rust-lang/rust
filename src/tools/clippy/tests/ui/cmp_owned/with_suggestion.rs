#[warn(clippy::cmp_owned)]
#[allow(clippy::unnecessary_operation, clippy::no_effect, unused_must_use, clippy::eq_op)]
fn main() {
    fn with_to_string(x: &str) {
        x != "foo".to_string();

        "foo".to_string() != x;
    }

    let x = "oh";

    with_to_string(x);

    x != "foo".to_owned();

    x != String::from("foo");

    42.to_string() == "42";

    Foo.to_owned() == Foo;

    "abc".chars().filter(|c| c.to_owned() != 'X');

    "abc".chars().filter(|c| *c != 'X');
}

struct Foo;

impl PartialEq for Foo {
    // Allow this here, because it emits the lint
    // without a suggestion. This is tested in
    // `tests/ui/cmp_owned/without_suggestion.rs`
    #[allow(clippy::cmp_owned)]
    fn eq(&self, other: &Self) -> bool {
        self.to_owned() == *other
    }
}

impl ToOwned for Foo {
    type Owned = Bar;
    fn to_owned(&self) -> Bar {
        Bar
    }
}

#[derive(PartialEq, Eq)]
struct Bar;

impl PartialEq<Foo> for Bar {
    fn eq(&self, _: &Foo) -> bool {
        true
    }
}

impl std::borrow::Borrow<Foo> for Bar {
    fn borrow(&self) -> &Foo {
        static FOO: Foo = Foo;
        &FOO
    }
}

#[derive(PartialEq, Eq)]
struct Baz;

impl ToOwned for Baz {
    type Owned = Baz;
    fn to_owned(&self) -> Baz {
        Baz
    }
}
