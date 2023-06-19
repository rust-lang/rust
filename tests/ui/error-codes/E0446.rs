struct Bar;
trait PrivTr {}

pub trait PubTr {
    type Alias1;
    type Alias2;
}

impl PubTr for u8 {
    type Alias1 = Bar; //~ ERROR E0446
    type Alias2 = Box<dyn PrivTr>; //~ ERROR E0446
}

fn main() {}
