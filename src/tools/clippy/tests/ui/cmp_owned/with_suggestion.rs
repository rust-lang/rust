#[warn(clippy::cmp_owned)]
#[allow(clippy::unnecessary_operation, clippy::no_effect, unused_must_use, clippy::eq_op)]
fn main() {
    fn with_to_string(x: &str) {
        x != "foo".to_string();
        //~^ cmp_owned

        "foo".to_string() != x;
        //~^ cmp_owned
    }

    let x = "oh";

    with_to_string(x);

    x != "foo".to_owned();
    //~^ cmp_owned

    x != String::from("foo");
    //~^ cmp_owned

    42.to_string() == "42";

    Foo.to_owned() == Foo;
    //~^ cmp_owned

    "abc".chars().filter(|c| c.to_owned() != 'X');
    //~^ cmp_owned

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

fn issue_8103() {
    let foo1 = String::from("foo");
    let _ = foo1 == "foo".to_owned();
    //~^ cmp_owned
    let foo2 = "foo";
    let _ = foo1 == foo2.to_owned();
    //~^ cmp_owned
}

macro_rules! issue16322_macro_generator {
    ($locale:ident) => {
        mod $locale {
            macro_rules! _make {
                ($token:tt) => {
                    stringify!($token)
                };
            }

            pub(crate) use _make;
        }

        macro_rules! t {
            ($token:tt) => {
                crate::$locale::_make!($token)
            };
        }
    };
}

issue16322_macro_generator!(de);

fn issue16322(item: String) {
    if item == t!(frohes_neu_Jahr).to_string() {
        //~^ cmp_owned
        println!("Ja!");
    }
}

fn issue16458() {
    macro_rules! partly_comes_from_macro {
        ($i:ident: $ty:ty, $def:expr) => {
            let _ = {
                let res = <$ty>::default() == $def;
                let _i: $ty = $def;
                res
            };
        };
    }

    partly_comes_from_macro! {
        required_version: String, env!("HOME").to_string()
    }

    macro_rules! all_comes_from_macro {
        ($($i:ident: $ty:ty, $def:expr);+ $(;)*) => {
            $(
                let _ = {
                    let res = <$ty>::default() == "$def".to_string();
                    //~^ cmp_owned
                    let _i: $ty = $def;
                    res
                };
            )+
        };
    }
    all_comes_from_macro! {
        required_version: String, env!("HOME").to_string();
    }
}
