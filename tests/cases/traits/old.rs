pub trait Abc {
    type A;
    fn test(&self) -> u8;
    fn test2(&self) -> u8;
    fn test4() -> u8 {
        0
    }
    fn test5() -> u8 {
        0
    }
    fn test7() -> u8;
    fn test8(&self) -> u8;
    fn test9(_: &Self) -> u8;
}
