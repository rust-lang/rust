#![feature(macro_metavar_expr)]
pub const IDX: usize = 1;

macro_rules! _direct_usage_super_2 {
    () => {
        macro_rules! _direct_usage_sub_2 {
            () => {
                $$crate
                //~^ ERROR the `$$crate` meta-variable expression is unstable
            }
        }
    };
}

macro_rules! indirect_usage_crate {
    ($d:tt) => {
        const _FOO: usize = $d$d crate::IDX;
        //~^ ERROR expected expression, found `$`
    };
}
macro_rules! indirect_usage_use {
    ($d:tt) => {
        indirect_usage_crate!($d);
    }
}
indirect_usage_use!($);

fn main() {}
