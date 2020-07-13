pub struct FirstStruct;

#[macro_export]
macro_rules! outer_macro {
    ($name:ident) => {
        #[macro_export]
        macro_rules! inner_macro {
            ($wrapper:ident) => {
                $wrapper!($name)
            }
        }
    }
}

outer_macro!(FirstStruct);
