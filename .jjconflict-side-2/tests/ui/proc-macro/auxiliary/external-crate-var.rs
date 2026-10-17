pub struct ExternFoo;

pub trait ExternTrait {
    const CONST: u32;
    type Assoc;
}

impl ExternTrait for ExternFoo {
    const CONST: u32 = 0;
    type Assoc = ExternFoo;
}

#[macro_export]
macro_rules! external { () => {
    mod bar {
        #[derive(Double)]
        struct Bar($crate::ExternFoo);
    }

    mod qself {
        #[derive(Double)]
        struct QSelf(<$crate::ExternFoo as $crate::ExternTrait>::Assoc);
    }

    mod qself_recurse {
        #[derive(Double)]
        struct QSelfRecurse(<
            <$crate::ExternFoo as $crate::ExternTrait>::Assoc
            as $crate::ExternTrait>::Assoc
        );
    }

    mod qself_in_const {
        #[derive(Double)]
        #[repr(u32)]
        enum QSelfInConst {
            Variant = <$crate::ExternFoo as $crate::ExternTrait>::CONST,
        }
    }
} }
