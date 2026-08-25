//@ known-bug: #148632
trait D<C> {}

trait Project {
    const SELF: dyn D<Self>;
}

fn main() {
    let _: &dyn Project<SELF = { 0 }>;
}
