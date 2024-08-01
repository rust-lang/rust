#![crate_name = "aCrate"]

mod a_module {
    pub fn private_function() {}

    pub use a_module::private_function as other_private_function;

    pub mod a_nested_module {
        //@ has aCrate/a_nested_module/index.html '//a[@href="fn.a_nested_public_function.html"]' 'a_nested_public_function'
        //@ hasraw aCrate/a_nested_module/fn.a_nested_public_function.html 'pub fn a_nested_public_function()'
        pub fn a_nested_public_function() {}

        //@ has aCrate/a_nested_module/index.html '//a[@href="fn.another_nested_public_function.html"]' 'another_nested_public_function'
        //@ hasraw aCrate/a_nested_module/fn.another_nested_public_function.html 'pub fn another_nested_public_function()'
        pub use a_nested_module::a_nested_public_function as another_nested_public_function;
    }

    //@ !hasraw aCrate/a_nested_module/index.html 'yet_another_nested_public_function'
    pub use a_nested_module::a_nested_public_function as yet_another_nested_public_function;

    //@ !hasraw aCrate/a_nested_module/index.html 'one_last_nested_public_function'
    pub use a_nested_module::another_nested_public_function as one_last_nested_public_function;
}

//@ !hasraw aCrate/index.html 'a_module'
//@ has aCrate/index.html '//a[@href="a_nested_module/index.html"]' 'a_nested_module'
pub use a_module::a_nested_module;

//@ has aCrate/index.html '//a[@href="fn.a_nested_public_function.html"]' 'a_nested_public_function'
//@ has aCrate/index.html '//a[@href="fn.another_nested_public_function.html"]' 'another_nested_public_function'
//@ has aCrate/index.html '//a[@href="fn.yet_another_nested_public_function.html"]' 'yet_another_nested_public_function'
//@ has aCrate/index.html '//a[@href="fn.one_last_nested_public_function.html"]' 'one_last_nested_public_function'
pub use a_module::{
    a_nested_module::{a_nested_public_function, another_nested_public_function},
    one_last_nested_public_function, yet_another_nested_public_function,
};

//@ has aCrate/index.html '//a[@href="fn.private_function.html"]' 'private_function'
//@ !hasraw aCrate/fn.private_function.html 'a_module'
//@ has aCrate/index.html '//a[@href="fn.other_private_function.html"]' 'other_private_function'
//@ !hasraw aCrate/fn.other_private_function.html 'a_module'
pub use a_module::{other_private_function, private_function};
