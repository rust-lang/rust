// error-pattern:expected `int` but found `bool`

fn main() {

    let a = {foo: 0i};

    let b = {foo: true with a};
}
