//@ edition:2021

trait Trait {}

struct IceCream;

impl IceCream {
    fn foo(_: &Trait) {}
    //~^ ERROR: expected a type, found a trait

    fn bar(self, _: &'a Trait) {}
    //~^ ERROR: expected a type, found a trait
    //~| ERROR: use of undeclared lifetime name

    fn alice<'a>(&self, _: &Trait) {}
    //~^ ERROR: expected a type, found a trait

    fn bob<'a>(_: &'a Trait) {}
    //~^ ERROR: expected a type, found a trait

    fn cat() -> &Trait {
        //~^ ERROR: missing lifetime specifier
        //~| ERROR: expected a type, found a trait
        &Type
    }

    fn dog<'a>() -> &Trait {
        //~^ ERROR: missing lifetime specifier
        //~| ERROR: expected a type, found a trait
        &Type
    }

    fn kitten() -> &'a Trait {
        //~^ ERROR: use of undeclared lifetime name
        //~| ERROR: expected a type, found a trait
        &Type
    }

    fn puppy<'a>() -> &'a Trait {
        //~^ ERROR: expected a type, found a trait
        &Type
    }

    fn parrot() -> &mut Trait {
        //~^ ERROR: missing lifetime specifier
        //~| ERROR: expected a type, found a trait
        &mut Type
    }
}

trait Sing {
    fn foo(_: &Trait);
    //~^ ERROR: expected a type, found a trait

    fn bar(_: &'a Trait);
    //~^ ERROR: expected a type, found a trait
    //~| ERROR: use of undeclared lifetime name

    fn alice<'a>(_: &Trait);
    //~^ ERROR: expected a type, found a trait

    fn bob<'a>(_: &'a Trait);
    //~^ ERROR: expected a type, found a trait

    fn cat() -> &Trait;
        //~^ ERROR: missing lifetime specifier
        //~| ERROR: expected a type, found a trait

    fn dog<'a>() -> &Trait {
        //~^ ERROR: missing lifetime specifier
        //~| ERROR: expected a type, found a trait
        &Type
    }

    fn kitten() -> &'a Trait {
        //~^ ERROR: use of undeclared lifetime name
        //~| ERROR: expected a type, found a trait
        &Type
    }

    fn puppy<'a>() -> &'a Trait {
        //~^ ERROR: expected a type, found a trait
        &Type
    }

    fn parrot() -> &mut Trait {
        //~^ ERROR: missing lifetime specifier
        //~| ERROR: expected a type, found a trait
        &mut Type
    }
}

fn foo(_: &Trait) {}
//~^ ERROR: expected a type, found a trait

fn bar(_: &'a Trait) {}
//~^ ERROR: expected a type, found a trait
//~| ERROR: use of undeclared lifetime name

fn alice<'a>(_: &Trait) {}
//~^ ERROR: expected a type, found a trait

fn bob<'a>(_: &'a Trait) {}
//~^ ERROR: expected a type, found a trait

struct Type;

impl Trait for Type {}

fn cat() -> &Trait {
//~^ ERROR: missing lifetime specifier
//~| ERROR: expected a type, found a trait
    &Type
}

fn dog<'a>() -> &Trait {
//~^ ERROR: missing lifetime specifier
//~| ERROR: expected a type, found a trait
    &Type
}

fn kitten() -> &'a Trait {
//~^ ERROR: use of undeclared lifetime name
//~| ERROR: expected a type, found a trait
    &Type
}

fn puppy<'a>() -> &'a Trait {
//~^ ERROR: expected a type, found a trait
    &Type
}

fn parrot() -> &mut Trait {
    //~^ ERROR: missing lifetime specifier
    //~| ERROR: expected a type, found a trait
    &mut Type
}

fn main() {}
