#[derive(Debug)]
pub enum EntryPointType {
    None,
    MainNamed,
    RustcMainAttr,
    Start,
    OtherMain, // Not an entry point, but some other function named main
}
