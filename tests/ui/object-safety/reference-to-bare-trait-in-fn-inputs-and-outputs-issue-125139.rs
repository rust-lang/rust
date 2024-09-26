//@ edition:2021

trait Trait {}

struct IceCream;

impl IceCream {
    fn foo(_: &Trait) {}
    //~^ ERROR: trait objects must include the `dyn` keyword

    fn bar(self, _: &'a Trait) {}
    //~^ ERROR: trait objects must include the `dyn` keyword
    //~| ERROR: use of undeclared lifetime name

    fn alice<'a>(&self, _: &Trait) {}
    //~^ ERROR: trait objects must include the `dyn` keyword

    fn bob<'a>(_: &'a Trait) {}
    //~^ ERROR: trait objects must include the `dyn` keyword

    fn cat() -> &Trait {
        //~^ ERROR: missing lifetime specifier
        //~| ERROR: trait objects must include the `dyn` keyword
        &Type
    }

    fn dog<'a>() -> &Trait {
        //~^ ERROR: missing lifetime specifier
        //~| ERROR: trait objects must include the `dyn` keyword
        &Type
    }

    fn kitten() -> &'a Trait {
        //~^ ERROR: use of undeclared lifetime name
        //~| ERROR: trait objects must include the `dyn` keyword
        &Type
    }

    fn puppy<'a>() -> &'a Trait {
        //~^ ERROR: trait objects must include the `dyn` keyword
        &Type
    }

    fn parrot() -> &mut Trait {
        //~^ ERROR: missing lifetime specifier
        //~| ERROR: trait objects must include the `dyn` keyword
        &mut Type
        //~^ ERROR: cannot return reference to temporary value
    }
}

trait Sing {
    fn foo(_: &Trait);
    //~^ ERROR: trait objects must include the `dyn` keyword

    fn bar(_: &'a Trait);
    //~^ ERROR: trait objects must include the `dyn` keyword
    //~| ERROR: use of undeclared lifetime name

    fn alice<'a>(_: &Trait);
    //~^ ERROR: trait objects must include the `dyn` keyword

    fn bob<'a>(_: &'a Trait);
    //~^ ERROR: trait objects must include the `dyn` keyword

    fn cat() -> &Trait;
        //~^ ERROR: missing lifetime specifier
        //~| ERROR: trait objects must include the `dyn` keyword

    fn dog<'a>() -> &Trait {
        //~^ ERROR: missing lifetime specifier
        //~| ERROR: trait objects must include the `dyn` keyword
        &Type
    }

    fn kitten() -> &'a Trait {
        //~^ ERROR: use of undeclared lifetime name
        //~| ERROR: trait objects must include the `dyn` keyword
        &Type
    }

    fn puppy<'a>() -> &'a Trait {
        //~^ ERROR: trait objects must include the `dyn` keyword
        &Type
    }

    fn parrot() -> &mut Trait {
        //~^ ERROR: missing lifetime specifier
        //~| ERROR: trait objects must include the `dyn` keyword
        &mut Type
        //~^ ERROR: cannot return reference to temporary value
    }
}

fn foo(_: &Trait) {}
//~^ ERROR: trait objects must include the `dyn` keyword

fn bar(_: &'a Trait) {}
//~^ ERROR: trait objects must include the `dyn` keyword
//~| ERROR: use of undeclared lifetime name

fn alice<'a>(_: &Trait) {}
//~^ ERROR: trait objects must include the `dyn` keyword

fn bob<'a>(_: &'a Trait) {}
//~^ ERROR: trait objects must include the `dyn` keyword

struct Type;

impl Trait for Type {}

fn cat() -> &Trait {
//~^ ERROR: missing lifetime specifier
//~| ERROR: trait objects must include the `dyn` keyword
    &Type
}

fn dog<'a>() -> &Trait {
//~^ ERROR: missing lifetime specifier
//~| ERROR: trait objects must include the `dyn` keyword
    &Type
}

fn kitten() -> &'a Trait {
//~^ ERROR: use of undeclared lifetime name
//~| ERROR: trait objects must include the `dyn` keyword
    &Type
}

fn puppy<'a>() -> &'a Trait {
//~^ ERROR: trait objects must include the `dyn` keyword
    &Type
}

fn parrot() -> &mut Trait {
    //~^ ERROR: missing lifetime specifier
    //~| ERROR: trait objects must include the `dyn` keyword
    &mut Type
    //~^ ERROR: cannot return reference to temporary value
}

fn main() {}
