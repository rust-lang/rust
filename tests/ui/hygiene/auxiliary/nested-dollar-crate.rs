pub const IN_DEF_CRATE: &str = "In def crate!";

macro_rules! make_it {
    () => {
        #[macro_export]
        macro_rules! inner {
            () => {
                $crate::IN_DEF_CRATE
            }
        }
    }
}

make_it!();
