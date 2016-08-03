// rustfmt-space_before_bound: true

trait Trait {}
fn f<'a, 'b : 'a, T : Trait>() {}
