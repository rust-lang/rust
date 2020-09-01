pub enum EntryPointType {
    None,
    MainNamed,
    MainAttr,
    Start,
    OtherMain, // Not an entry point, but some other function named main
}
