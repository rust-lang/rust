//@ edition:2021

#[macro_export]
macro_rules! m {
    ($expr:expr) => {
        compile_error!("did not expect an expression to be parsed");
    };
    (const { }) => {};
}
