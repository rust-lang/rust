// Confusing diagnostic when using variable as a type:
//
// Previous warnings indicate Foo is not used, when in fact it is
// used improperly as a variable or constant. New warning points
// out user may be trying to use variable as a type. Test demonstrates
// cases for both local variable and const.

fn main() {
    let Baz: &str = "";

    println!("{}", Baz::Bar); //~ ERROR: failed to resolve: use of undeclared type `Baz`
}

#[allow(non_upper_case_globals)]
pub const Foo: &str = "";

mod submod {
    use super::Foo;
    fn function() {
        println!("{}", Foo::Bar); //~ ERROR: failed to resolve: use of undeclared type `Foo`
    }
}
