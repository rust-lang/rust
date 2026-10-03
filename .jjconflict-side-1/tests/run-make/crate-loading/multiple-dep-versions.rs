extern crate dep_2_reexport;
extern crate dependency;
use dep_2_reexport::{Error2, OtherType, Trait2, Type};
use dependency::{Error, OtherError, Trait, do_something, do_something_trait, do_something_type};

fn main() -> Result<(), Error> {
    do_something(Type);
    Type.foo();
    Type::bar();
    do_something(OtherType);
    do_something_type(Type);
    do_something_trait(Box::new(Type) as Box<dyn Trait2>);
    Err(Error2)?;
    Ok(())
}

fn foo() -> Result<(), OtherError> {
    Err(Error2)?;
    Ok(())
}
