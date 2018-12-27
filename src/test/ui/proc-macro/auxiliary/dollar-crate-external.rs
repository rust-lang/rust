pub type S = u8;

#[macro_export]
macro_rules! external {
    () => {
        dollar_crate::m! {
            struct M($crate::S);
        }

        #[dollar_crate::a]
        struct A($crate::S);

        #[derive(dollar_crate::d)]
        struct D($crate::S);
    };
}
