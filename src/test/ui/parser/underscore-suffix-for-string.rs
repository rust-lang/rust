// build-pass (FIXME(62277): could be check-pass?)

fn main() {
    let _ = "Foo"_;
    //~^ WARNING underscore literal suffix is not allowed
    //~| WARNING this was previously accepted
    //~| NOTE issue #42326
}
