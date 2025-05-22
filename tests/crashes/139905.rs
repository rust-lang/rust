//@ known-bug: #139905
trait a<const b: bool> {}
impl a<{}> for () {}
trait c {}
impl<const d: u8> c for () where (): a<d> {}
impl c for () {}
