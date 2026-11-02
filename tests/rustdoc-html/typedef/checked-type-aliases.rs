#![feature(checked_type_aliases)]
#![expect(incomplete_features)]
#![crate_name = "it"]

//@ has it/type.Alias.html '//pre[@class="rust item-decl"]' \
//          "type Alias<T: Iterator> = (T, i32) where String: From<T>;"
pub type Alias<T: Iterator> = (T, i32)
where
    String: From<T>;
