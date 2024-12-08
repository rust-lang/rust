pub fn foo() {}

#[macro_export]
macro_rules! gimme_a {
    ($($mac:tt)*) => { $($mac)* { 'a } }
}
