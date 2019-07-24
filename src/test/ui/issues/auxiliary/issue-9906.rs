pub use other::FooBar;
pub use other::foo;

mod other {
    pub struct FooBar{value: isize}
    impl FooBar{
        pub fn new(val: isize) -> FooBar {
            FooBar{value: val}
        }
    }

    pub fn foo(){
        let _ = 1 + 1;
    }
}
