//@ run-pass

#![feature(decl_macro)]
#![allow(non_local_definitions)]

trait Value {
    fn value(self) -> &'static str;
}

const _: () = {
    macro implement() {
        fn helper() -> &'static str {
            "generated"
        }

        impl Value for () {
            fn value(self) -> &'static str {
                helper()
            }
        }
    }

    implement!();
};

fn generated_items() {
    fn helper() -> &'static str {
        "caller"
    }

    macro generate($name:ident, $value:expr) {
        fn helper() -> &'static str {
            $value
        }

        fn $name() -> &'static str {
            helper()
        }
    }

    generate!(first, "first expansion");
    generate!(second, "second expansion");
    assert_eq!(first(), "first expansion");
    assert_eq!(second(), "second expansion");
    assert_eq!(helper(), "caller");

    {
        generate!(nested, "nested expansion");
        assert_eq!(nested(), "nested expansion");
    }
}

fn definition_site_locals() {
    let value = "definition-site";
    macro read_value() {
        value
    }
    let value = "shadowing";
    assert_eq!(read_value!(), "definition-site");
    assert_eq!(value, "shadowing");

    let legacy = "definition-site";
    macro_rules! read_legacy {
        () => {
            legacy
        };
    }
    let legacy = "shadowing";
    assert_eq!(read_legacy!(), "definition-site");
    assert_eq!(legacy, "shadowing");
}

fn definition_site_items() {
    fn helper() -> &'static str {
        "definition-site"
    }

    macro call_helper() {
        helper()
    }

    assert_eq!(call_helper!(), "definition-site");
    {
        fn helper() -> &'static str {
            "shadowing"
        }

        assert_eq!(call_helper!(), "definition-site");
        assert_eq!(helper(), "shadowing");
    }
}

fn main() {
    assert_eq!(().value(), "generated");
    generated_items();
    definition_site_locals();
    definition_site_items();
}
