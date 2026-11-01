// Empty predicate list is OK
fn equal1<T>(_: &T, _: &T) -> bool where {
    true
}

// Empty bound list is OK
fn equal2<T>(_: &T, _: &T) -> bool where T: {
    true
}

fn foo<'a>() where 'a {}
//~^ ERROR expected `:`, found `{`

fn main() {
}
