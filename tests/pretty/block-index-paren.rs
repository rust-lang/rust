//@ pretty-mode:expanded
//@ pp-exact:block-index-paren.pp

macro_rules! block_arr {
    () => {{ [0u8; 4] }};
}

macro_rules! as_slice {
    () => {{ &block_arr!()[..] }};
}

fn main() { let _: &[u8] = as_slice!(); }
