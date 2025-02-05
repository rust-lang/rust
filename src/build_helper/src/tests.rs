#[test]
fn string_enums() {
    // These imports are needed for the macro-generated code
    use std::fmt;
    use std::str::FromStr;

    crate::compiletest::string_enum! {
        #[derive(Clone, Copy, Debug, PartialEq)]
        enum Animal {
            Cat => "meow",
            Dog => "woof",
        }
    }

    // General assertions, mostly to silence the dead code warnings
    assert_eq!(Animal::VARIANTS.len(), 2);
    assert_eq!(Animal::STR_VARIANTS.len(), 2);

    // Correct string conversions
    assert_eq!(Animal::Cat, "meow".parse().unwrap());
    assert_eq!(Animal::Dog, "woof".parse().unwrap());

    // Invalid conversions
    let animal = "nya".parse::<Animal>();
    assert_eq!("unknown `Animal` variant: `nya`", animal.unwrap_err());
}
