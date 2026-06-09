//@ check-pass

struct Sqlite {}

trait HasArguments<'q> {
    type Arguments;
}

impl<'q> HasArguments<'q> for Sqlite {
    type Arguments = std::marker::PhantomData<&'q ()>;
}

fn foo() {
    let _ = <Sqlite as HasArguments>::Arguments::default();
}

fn main() {}
