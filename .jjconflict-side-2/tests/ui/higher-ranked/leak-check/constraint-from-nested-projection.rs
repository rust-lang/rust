//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@[old] check-pass

// Evaluate in the old solver does apply constraints from nested `Projection` obligations,
// as they can constrain otherwise unconstrained inference variables. This also allows
// `Projection` goals to otherwise influence its parent obligation by returning constraints
// from matching the impl header.
//
// The new solver entirely ignores region constraints from nested goals in the leak check.
// This is the one case where the the implementation of the new solver will actually weaken
// the leak check.

trait ProjectStatic<'a> {
    type Assoc;
}
impl ProjectStatic<'static> for () {
    type Assoc = u32;
}

trait RequiresProject<'a> {}
impl<'a, T: ProjectStatic<'a, Assoc = u32>> RequiresProject<'a> for T {}

trait Pick<T> {}
impl<T: for<'a> RequiresProject<'a>> Pick<u32> for T {}
impl<T> Pick<u16> for T {}

fn pick<T: Pick<U>, U>() {}

fn main() {
    pick::<(), _>();
    //[next]~^ ERROR: type annotations needed
}
