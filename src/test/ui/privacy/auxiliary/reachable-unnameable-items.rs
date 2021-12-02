use inner_private_module::*;

mod inner_private_module {
    pub struct Unnameable1;
    pub struct Unnameable2;
    #[derive(Clone, Copy)]
    pub struct Unnameable3;
    pub struct Unnameable4;
    pub struct Unnameable5;
    pub struct Unnameable6;
    pub struct Unnameable7;
    #[derive(Default)]
    pub struct Unnameable8;
    pub enum UnnameableEnum {
        NameableVariant
    }
    pub trait UnnameableTrait {
        type Alias: Default;
    }

    impl Unnameable1 {
        pub fn method_of_unnameable_type1(&self) -> &'static str {
            "Hello1"
        }
    }
    impl Unnameable2 {
        pub fn method_of_unnameable_type2(&self) -> &'static str {
            "Hello2"
        }
    }
    impl Unnameable3 {
        pub fn method_of_unnameable_type3(&self) -> &'static str {
            "Hello3"
        }
    }
    impl Unnameable4 {
        pub fn method_of_unnameable_type4(&self) -> &'static str {
            "Hello4"
        }
    }
    impl Unnameable5 {
        pub fn method_of_unnameable_type5(&self) -> &'static str {
            "Hello5"
        }
    }
    impl Unnameable6 {
        pub fn method_of_unnameable_type6(&self) -> &'static str {
            "Hello6"
        }
    }
    impl Unnameable7 {
        pub fn method_of_unnameable_type7(&self) -> &'static str {
            "Hello7"
        }
    }
    impl Unnameable8 {
        pub fn method_of_unnameable_type8(&self) -> &'static str {
            "Hello8"
        }
    }
    impl UnnameableEnum {
        pub fn method_of_unnameable_enum(&self) -> &'static str {
            "HelloEnum"
        }
    }
}

pub fn function_returning_unnameable_type() -> Unnameable1 {
    Unnameable1
}

pub const CONSTANT_OF_UNNAMEABLE_TYPE: Unnameable2 =
                                            Unnameable2;

pub fn function_accepting_unnameable_type(_: Option<Unnameable3>) {}

pub type AliasOfUnnameableType = Unnameable4;

impl Unnameable1 {
    pub fn inherent_method_returning_unnameable_type(&self) -> Unnameable5 {
        Unnameable5
    }
}

pub trait Tr {
    fn trait_method_returning_unnameable_type(&self) -> Unnameable6 {
        Unnameable6
    }
}
impl Tr for Unnameable1 {}

pub use inner_private_module::UnnameableEnum::NameableVariant;

pub struct Struct {
    pub field_of_unnameable_type: Unnameable7
}

pub static STATIC: Struct = Struct { field_of_unnameable_type: Unnameable7 } ;

impl UnnameableTrait for AliasOfUnnameableType {
    type Alias = Unnameable8;
}

pub fn generic_function<T: UnnameableTrait>() -> T::Alias {
    Default::default()
}
