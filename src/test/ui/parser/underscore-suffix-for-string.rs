// check-fail

// njn: not sure about the change to this one...

fn main() {
    let _ = "Foo"_;
    //~^ ERROR suffixes on string literals are invalid
    //~| NOTE invalid suffix `_`
    //~^^^ WARNING underscore literal suffix is not allowed
    //~| WARNING this was previously accepted
    //~| NOTE issue #42326
}
