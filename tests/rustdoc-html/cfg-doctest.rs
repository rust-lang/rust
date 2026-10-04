//@ !has cfg_doctest/type.SomeStruct.html
//@ !has cfg_doctest/index.html '//a/@href' 'type.SomeStruct.html'

/// Sneaky, this isn't actually part of docs.
#[cfg(doctest)]
pub struct SomeStruct;
