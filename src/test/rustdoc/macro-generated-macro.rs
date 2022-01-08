macro_rules! make_macro {
    ($macro_name:ident $($matcher:tt)*) => {
        #[macro_export]
        macro_rules! $macro_name {
            (<= $($matcher)* =>) => {};
        }
    }
}

// @has macro_generated_macro/macro.interpolations.html //pre 'macro_rules! interpolations {'
// @has - //pre '(<= type $($i : ident) :: * + $e : expr =>) => { ... };'
make_macro!(interpolations type $($i:ident)::* + $e:expr);
interpolations!(<= type foo::bar + x.sort() =>);

// @has macro_generated_macro/macro.attributes.html //pre 'macro_rules! attributes {'
// @has - //pre '(<= #! [no_std] #[inline] =>) => { ... };'
make_macro!(attributes #![no_std] #[inline]);

// @has macro_generated_macro/macro.groups.html //pre 'macro_rules! groups {'
// @has - //pre '(<= fn {} () { foo [0] } =>) => { ... };'
make_macro!(groups fn {}() {foo[0]});
