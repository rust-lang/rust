fn main() {
    let _ = const {
        //~^ ERROR expected expression, found keyword `const`
        true
    };
}
