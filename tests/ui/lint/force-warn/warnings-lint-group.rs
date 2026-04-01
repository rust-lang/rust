// --force-warn warnings is an error
//@ compile-flags: --force-warn warnings

fn main() {}

//~? ERROR `warnings` lint group is not supported with ´--force-warn´
//~? ERROR `warnings` lint group is not supported with ´--force-warn´
//~? ERROR `warnings` lint group is not supported with ´--force-warn´
