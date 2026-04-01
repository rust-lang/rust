#![crate_type = "lib"]
#![deny(unknown_or_malformed_diagnostic_attributes)]


#[diagnostic::on_unimplemented(message = "here is a big \
                                         multiline string \
                                         {unknown}")]
//~^ ERROR there is no parameter `unknown` on trait `MultiLine` [malformed_diagnostic_format_literals]
pub trait MultiLine {}

#[diagnostic::on_unimplemented(message = "here is a big \
                                         multiline string {unknown}")]
//~^ ERROR there is no parameter `unknown` on trait `MultiLine2` [malformed_diagnostic_format_literals]
pub trait MultiLine2 {}

#[diagnostic::on_unimplemented(message = "here is a big \
    multiline string {unknown}")]
//~^ ERROR there is no parameter `unknown` on trait `MultiLine3` [malformed_diagnostic_format_literals]
pub trait MultiLine3 {}


#[diagnostic::on_unimplemented(message = "here is a big \
\
                \
                                \
                                                \
    multiline string {unknown}")]
//~^ ERROR there is no parameter `unknown` on trait `MultiLine4` [malformed_diagnostic_format_literals]
pub trait MultiLine4 {}

#[diagnostic::on_unimplemented(message = "here is a big \
                                         multiline string \
                                         {Self:+}")]
//~^ ERROR invalid format specifier [malformed_diagnostic_format_literals]
pub trait MultiLineFmt {}

#[diagnostic::on_unimplemented(message = "here is a big \
                                         multiline string {Self:X}")]
//~^ ERROR invalid format specifier [malformed_diagnostic_format_literals]
pub trait MultiLineFmt2 {}

#[diagnostic::on_unimplemented(message = "here is a big \
    multiline string {Self:#}")]
//~^ ERROR invalid format specifier [malformed_diagnostic_format_literals]
pub trait MultiLineFmt3 {}


#[diagnostic::on_unimplemented(message = "here is a big \
\
                \
                                \
                                                \
    multiline string {Self:?}")]
//~^ ERROR invalid format specifier [malformed_diagnostic_format_literals]
pub trait MultiLineFmt4 {}
