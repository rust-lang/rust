#[macro_export]
macro_rules! my_file {
    () => { file!() }
}

pub fn file() -> &'static str {
    file!()
}
