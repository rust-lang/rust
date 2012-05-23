// error-pattern: copying a noncopyable value

resource r(i: @mut int) {
    *i = *i + 1;
}

fn main() {
    let i = @mut 0;
    {
        // Can't do this copy
        let x = ~~~{y: r(i)};
        let z = x;
        log(debug, x);
    }
    log(error, *i);
}