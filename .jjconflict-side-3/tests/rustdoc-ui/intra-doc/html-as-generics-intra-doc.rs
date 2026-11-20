#![deny(rustdoc::invalid_html_tags)]
#![deny(rustdoc::broken_intra_doc_links)]

pub struct ExistentStruct<T>(T);

/// This [test][ExistentStruct<i32>] thing!
pub struct NoError;

/// This [ExistentStruct<i32>] thing!
//~^ ERROR unclosed HTML tag `i32`
pub struct PartialErrorOnlyHtml;

/// This [test][NonExistentStruct<i32>] thing!
//~^ ERROR unresolved link
pub struct PartialErrorOnlyResolve;

/// This [NonExistentStruct2<i32>] thing!
//~^ ERROR unclosed HTML tag `i32`
//~| ERROR unresolved link
pub struct YesError;

/// This [NonExistentStruct3<i32>][] thing!
//~^ ERROR unclosed HTML tag `i32`
//~| ERROR unresolved link
pub struct YesErrorCollapsed;
