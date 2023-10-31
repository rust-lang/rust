// FIXME(aliemjay): this should not pass.
// check-pass

trait Trait {
    type Ty<'a> where Self: 'a;
}

impl Trait for () {
    type Ty<'a> = () where Self: 'a;
}

pub fn body<'a>() {
    // ill-formed projection..
    let _: Option<<() as Trait>::Ty::<'a>> = None;
}

fn main() {}
