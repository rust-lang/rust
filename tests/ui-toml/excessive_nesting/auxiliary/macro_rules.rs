#[rustfmt::skip]
#[macro_export]
macro_rules! excessive_nesting {
    () => {{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{
        println!("hi!!")
    }}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
}
