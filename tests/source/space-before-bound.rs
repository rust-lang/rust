// rustfmt-space_before_bound: true

trait Trait {}
trait Trait2 {}
fn f<'a, 'b: 'a, T: Trait, U>() where U: Trait2 {}

// should fit on the line
fn f2<'a, 'b: 'a, Ttttttttttttttttttttttttttttttttttttttttttttttt: Trait, U>() where U: Trait2 {}
// should be wrapped
fn f2<'a, 'b: 'a, Tttttttttttttttttttttttttttttttttttttttttttttttt: Trait, U>() where U: Trait2 {}
