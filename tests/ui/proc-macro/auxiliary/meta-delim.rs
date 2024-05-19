macro_rules! produce_it {
    ($dollar_one:tt $foo:ident $my_name:ident) => {
        #[macro_export]
        macro_rules! meta_delim {
            ($dollar_one ($dollar_one $my_name:ident)*) => {
                stringify!($dollar_one ($dollar_one $my_name)*)
            }
        }
    }
}

produce_it!($my_name name);
