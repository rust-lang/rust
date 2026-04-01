#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

type Alias = Local;
struct Local;

impl Alias { fn method() {} } //~ ERROR duplicate definitions with name `method`
impl Local { fn method() {} }

fn main() {}
