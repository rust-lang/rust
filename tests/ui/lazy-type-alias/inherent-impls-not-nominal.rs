#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

type Alias = <() as Trait>::Out;

trait Trait { type Out; }
impl Trait for () { type Out = Local; }
struct Local;

impl Alias {} //~ ERROR no nominal type found for inherent implementation

fn main() {}
