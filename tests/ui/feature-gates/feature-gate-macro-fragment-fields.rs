#![crate_type = "lib"]

macro_rules! m {
    //~v ERROR: macro fragment fields are unstable
    ($x:ident) => { ${x.field} };
}
