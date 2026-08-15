#[macro_export]
macro_rules! creator {
    (struct $name1:ident; enum $name2:ident; enum $name3:ident;) => {
        #[derive(Debug)]
        pub struct $name1;

        #[derive(Debug)]
        #[repr(u32)]
        pub enum $name2 { A }

        #[derive(Debug)]
        #[repr(u16)]
        pub enum $name3 { A }
    }
}
