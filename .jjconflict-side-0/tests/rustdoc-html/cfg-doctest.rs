//@ !has cfg_doctest/struct.SomeStruct.html
//@ !has cfg_doctest/index.html '//a/@href' 'struct.SomeStruct.html'

/// Sneaky, this isn't actually part of docs.
#[cfg(doctest)]
pub struct SomeStruct;
