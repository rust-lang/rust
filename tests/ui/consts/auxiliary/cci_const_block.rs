pub static BLOCK_FN_DEF: fn(usize) -> usize = {
    fn foo(a: usize) -> usize {
        a + 10
    }
    foo
};
