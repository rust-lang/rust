// rustfmt-space_before_bound: true
// rustfmt-space_after_bound_colon: false

trait Trait {}
fn f<'a, 'b :'a, T :Trait>() {}
