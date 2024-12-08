// Regression test for #83026.
// The goal of this test is to ensure that impl blocks inside
// const expressions are documented as well.

#![crate_name = "foo"]

//@ has 'foo/struct.A.html'
//@ has - '//*[@id="method.new"]/*[@class="code-header"]' 'pub fn new() -> A'
//@ has - '//*[@id="method.bar"]/*[@class="code-header"]' 'pub fn bar(&self)'
//@ has - '//*[@id="method.woo"]/*[@class="code-header"]' 'pub fn woo(&self)'
//@ has - '//*[@id="method.yoo"]/*[@class="code-header"]' 'pub fn yoo()'
//@ has - '//*[@id="method.yuu"]/*[@class="code-header"]' 'pub fn yuu()'
pub struct A;

const _: () = {
    impl A {
        const FOO: () = {
            impl A {
                pub fn woo(&self) {}
            }
        };

        pub fn new() -> A {
            A
        }
    }
};
pub const X: () = {
    impl A {
        pub fn bar(&self) {}
    }
};

fn foo() {
    impl A {
        pub fn yoo() {}
    }
    const _: () = {
        impl A {
            pub fn yuu() {}
        }
    };
}
