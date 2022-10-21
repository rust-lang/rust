#[macro_export]
macro_rules! foo {
    () => {
        let x: &Option<i32> = &Some(1);
        let _y = x as *const Option<i32>;
    }
}
