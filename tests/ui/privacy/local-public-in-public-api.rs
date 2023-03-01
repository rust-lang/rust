// This ui test checks that private types present in public APIs get
// a warning.

#![crate_type = "lib"]
#![deny(private_in_public)]

mod my_type {
    pub struct Field;
    pub struct MyType {
        pub field: Field,
    }

    impl MyType {
        pub fn builder() -> MyTypeBuilder {
                            //~^ ERROR
                            //~| WARN
            MyTypeBuilder { field: Field }
        }

        pub fn with_builder(_: MyTypeBuilder) {}
                               //~^ ERROR
                               //~| WARN
    }

    pub struct MyTypeBuilder {
        field: Field,
    }

    impl MyTypeBuilder {
        pub fn field(mut self, val: Field) -> Self {
            self.field = val;

            self
        }

        pub fn build(self) -> MyType {
            MyType { field: self.field }
        }
    }
}

pub use my_type::{MyType, Field};
