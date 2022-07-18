// All possible usages of $$crate are currently forbidden

pub const IDX: usize = 1;

macro_rules! _direct_usage_super {
    () => {
        macro_rules! _direct_usage_sub {
            () => {{
                $$crate
                //~^ ERROR unexpected token: crate
            }};
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

fn main() {
}
