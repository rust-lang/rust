#![warn(clippy::empty_enum_variants_with_brackets)]
#![allow(dead_code)]

pub enum PublicTestEnum {
    NonEmptyBraces { x: i32, y: i32 }, // No error
    NonEmptyParentheses(i32, i32),     // No error
    EmptyBraces {},
    //~^ empty_enum_variants_with_brackets
    EmptyParentheses(), // No error as enum is pub
}

enum TestEnum {
    NonEmptyBraces { x: i32, y: i32 }, // No error
    NonEmptyParentheses(i32, i32),     // No error
    EmptyBraces {},
    //~^ empty_enum_variants_with_brackets
    EmptyParentheses(),
    //~^ empty_enum_variants_with_brackets
    AnotherEnum, // No error
}

mod issue12551 {
    enum EvenOdd {
        // Used as functions -> no error
        Even(),
        Odd(),
        // Not used as a function
        Unknown(),
        //~^ empty_enum_variants_with_brackets
    }

    fn even_odd(x: i32) -> EvenOdd {
        (x % 2 == 0).then(EvenOdd::Even).unwrap_or_else(EvenOdd::Odd)
    }

    fn natural_number(x: i32) -> NaturalOrNot {
        (x > 0)
            .then(NaturalOrNot::Natural)
            .unwrap_or_else(NaturalOrNot::NotNatural)
    }

    enum NaturalOrNot {
        // Used as functions -> no error
        Natural(),
        NotNatural(),
        // Not used as a function
        Unknown(),
        //~^ empty_enum_variants_with_brackets
    }

    enum RedundantParenthesesFunctionCall {
        // Used as a function call but with redundant parentheses
        Parentheses(),
        //~^ empty_enum_variants_with_brackets
        // Not used as a function
        NoParentheses,
    }

    #[allow(clippy::no_effect)]
    fn redundant_parentheses_function_call() {
        // The parentheses in the below line are redundant.
        RedundantParenthesesFunctionCall::Parentheses();
        RedundantParenthesesFunctionCall::NoParentheses;
    }

    // Same test as above but with usage of the enum occurring before the definition.
    #[allow(clippy::no_effect)]
    fn redundant_parentheses_function_call_2() {
        // The parentheses in the below line are redundant.
        RedundantParenthesesFunctionCall2::Parentheses();
        RedundantParenthesesFunctionCall2::NoParentheses;
    }

    enum RedundantParenthesesFunctionCall2 {
        // Used as a function call but with redundant parentheses
        Parentheses(),
        //~^ empty_enum_variants_with_brackets
        // Not used as a function
        NoParentheses,
    }
}

enum TestEnumWithFeatures {
    NonEmptyBraces {
        #[cfg(feature = "thisisneverenabled")]
        x: i32,
    }, // No error
    NonEmptyParentheses(#[cfg(feature = "thisisneverenabled")] i32), // No error
}

#[derive(Clone)]
enum Foo {
    Variant1(i32),
    Variant2,
    Variant3(), //~ ERROR: enum variant has empty brackets
}

#[derive(Clone)]
pub enum PubFoo {
    Variant1(i32),
    Variant2,
    Variant3(),
}

fn main() {}
