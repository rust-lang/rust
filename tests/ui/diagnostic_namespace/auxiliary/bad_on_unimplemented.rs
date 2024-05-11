#[diagnostic::on_unimplemented(aa = "broken")]
pub trait MissingAttr {}

#[diagnostic::on_unimplemented(label = "a", label = "b")]
pub trait DuplicateAttr {}

#[diagnostic::on_unimplemented = "broken"]
pub trait NotMetaList {}

#[diagnostic::on_unimplemented]
pub trait Empty {}

#[diagnostic::on_unimplemented {}]
pub trait WrongDelim {}

#[diagnostic::on_unimplemented(label = "{A:.3}")]
pub trait BadFormatter<A> {}

#[diagnostic::on_unimplemented(label = "test {}")]
pub trait NoImplicitArgs {}

#[diagnostic::on_unimplemented(label = "{missing}")]
pub trait MissingArg {}

#[diagnostic::on_unimplemented(label = "{_}")]
pub trait BadArg {}
