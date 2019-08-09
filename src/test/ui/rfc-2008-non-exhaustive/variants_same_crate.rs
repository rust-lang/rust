// run-pass

#![feature(non_exhaustive)]

pub enum NonExhaustiveVariants {
    #[non_exhaustive] Unit,
    #[non_exhaustive] Tuple(u32),
    #[non_exhaustive] Struct { field: u32 }
}

fn main() {
    let variant_tuple = NonExhaustiveVariants::Tuple(340);
    let variant_struct = NonExhaustiveVariants::Struct { field: 340 };

    match variant_tuple {
        NonExhaustiveVariants::Unit => "",
        NonExhaustiveVariants::Tuple(fe_tpl) => "",
        NonExhaustiveVariants::Struct { field } => ""
    };
}
