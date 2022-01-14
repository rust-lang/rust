macro_rules! make_macro {
    ($macro_name:ident $($matcher:tt)*) => {
        #[macro_export]
        macro_rules! $macro_name {
            (<= $($matcher)* =>) => {};
        }
    }
}

// @has macro_generated_macro/macro.interpolations.html //pre 'macro_rules! interpolations {'
// @has - //pre '(<= type $($i:ident)::* + $e:expr =>) => { ... };'
make_macro!(interpolations type $($i:ident)::* + $e:expr);
interpolations!(<= type foo::bar + x.sort() =>);

// @has macro_generated_macro/macro.attributes.html //pre 'macro_rules! attributes {'
// @has - //pre '(<= #![no_std] #[cfg(feature = "alloc")] =>) => { ... };'
make_macro!(attributes #![no_std] #[cfg(feature = "alloc")]);

// @has macro_generated_macro/macro.groups.html //pre 'macro_rules! groups {'
// @has - //pre '(<= fn {} () { foo[0] } =>) => { ... };'
make_macro!(groups fn {}() {foo[0]});

// @snapshot macro_linebreak_pre macro_generated_macro/macro.linebreak.html //pre
make_macro!(linebreak 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28);
