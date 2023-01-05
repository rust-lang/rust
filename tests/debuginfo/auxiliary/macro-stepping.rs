// compile-flags:-g

#![crate_type = "rlib"]

#[macro_export]
macro_rules! new_scope {
    () => {
        let x = 1;
    }
}
