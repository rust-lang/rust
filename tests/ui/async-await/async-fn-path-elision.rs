//@ edition:2018

struct HasLifetime<'a>(&'a bool);

async fn error(lt: HasLifetime) { //~ ERROR implicit elided lifetime not allowed here
    if *lt.0 {}
}

fn no_error(lt: HasLifetime) {
    if *lt.0 {}
}

fn main() {}
