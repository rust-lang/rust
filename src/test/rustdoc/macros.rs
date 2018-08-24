// @has macros/macro.my_macro.html //pre 'macro_rules! my_macro {'
// @has - //pre '() => { ... };'
// @has - //pre '($a:tt) => { ... };'
// @has - //pre '($e:expr) => { ... };'
// @has macros/macro.my_macro!.html
// @has - //a 'macro.my_macro.html'
#[macro_export]
macro_rules! my_macro {
    () => [];
    ($a:tt) => ();
    ($e:expr) => {};
}
