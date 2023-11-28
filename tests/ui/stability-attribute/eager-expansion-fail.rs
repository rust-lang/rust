#![crate_type = "lib"]
#![feature(staged_api)]
#![stable(feature = "stable_test_feature", since = "3.3.3")]

macro_rules! not_a_literal {
    () => {
        boohoo
    };
}

macro_rules! m {
    () => {
        #[stable(feature = 1 + 1, since = "?")]
        //~^ expression in the value of this attribute must be a literal or macro call
        //~^^ incorrect meta item
        //~^^^ incorrect meta item
        pub struct Math; //~ struct has missing stability attribute

        #[stable(feature = not_a_literal!(), since = "?")]
        //~^ expression in the value of this attribute must be a literal or macro call
        //~^^ incorrect meta item
        //~^^^ incorrect meta item
        pub struct NotLiteral; //~ struct has missing stability attribute
    };
}

m!();
