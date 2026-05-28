fn main() {
    let _: Vec<u8, _> = vec![]; //~ ERROR use of unstable library feature `allocator_api`
    #[rustfmt::skip]
    let _: Vec<
        String,
        _> = vec![]; //~ ERROR use of unstable library feature `allocator_api`
    let _ = Vec::<u16, _>::new(); //~ ERROR use of unstable library feature `allocator_api`
    let _boxed: Box<u32, _> = Box::new(10); //~ ERROR use of unstable library feature `allocator_api`
}
