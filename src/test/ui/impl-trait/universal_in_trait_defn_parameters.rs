// run-pass

use std::fmt::Debug;

trait InTraitDefnParameters {
    fn in_parameters(_: impl Debug) -> String;
}

impl InTraitDefnParameters for () {
    fn in_parameters(v: impl Debug) -> String {
        format!("() + {:?}", v)
    }
}

fn main() {
    let s = <() as InTraitDefnParameters>::in_parameters(22);
    assert_eq!(s, "() + 22");
}
