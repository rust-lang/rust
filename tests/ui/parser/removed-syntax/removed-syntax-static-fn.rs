struct S;

impl S {
    static fn f() {}
    //~^ ERROR expected identifier, found keyword `fn`
    //~| ERROR expected one of `:`, `;`, or `=`
    //~| ERROR missing type for `static` item
}

fn main() {}
