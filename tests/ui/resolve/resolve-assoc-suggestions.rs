// Make sure associated items are recommended only in appropriate contexts.

struct S {
    field: u8,
}

trait Tr {
    fn method(&self);
    type Type;
}

impl Tr for S {
    type Type = u8;

    fn method(&self) {
        let _: field;
        //~^ ERROR cannot find type `field`
        let field(..);
        //~^ ERROR cannot find tuple struct or tuple variant `field`
        field;
        //~^ ERROR cannot find value `field`

        let _: Type;
        //~^ ERROR cannot find type `Type`
        let Type(..);
        //~^ ERROR cannot find tuple struct or tuple variant `Type`
        Type;
        //~^ ERROR cannot find value `Type`

        let _: method;
        //~^ ERROR cannot find type `method`
        let method(..);
        //~^ ERROR cannot find tuple struct or tuple variant `method`
        method;
        //~^ ERROR cannot find value `method`
    }
}

fn main() {}
