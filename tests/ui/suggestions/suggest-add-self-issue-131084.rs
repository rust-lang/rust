//@ check-fail

// A recovered `&param` should not make the missing-`self` suggestion insert
// the receiver after the leading `&`.

struct SomeStruct;

impl SomeStruct {
    fn some_fn(&some_name) {
        //~^ ERROR expected one of `:`, `@`, or `|`, found `)`
        self
        //~^ ERROR expected value, found module `self`
    }

    fn mut_param(mut some_name) {
        //~^ ERROR expected one of `:`, `@`, or `|`, found `)`
        self
        //~^ ERROR expected value, found module `self`
    }

    fn type_before_name(String s) {
        //~^ ERROR expected one of `:`, `@`, or `|`, found `s`
        self
        //~^ ERROR expected value, found module `self`
    }
}

fn main() {}
