// Regression test for #160231: `<wbr>` insertion sliced mid-character on an item name whose
// grapheme cluster starts with a `Prepend` character and contains `_`.

#![crate_name = "foo"]
#![allow(mixed_script_confusables, non_camel_case_types)]

//@ hasraw foo/index.html 'abcൎ_<wbr>defgh'
pub struct abcൎ_defgh;
