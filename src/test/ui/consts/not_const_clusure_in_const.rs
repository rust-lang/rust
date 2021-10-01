// run-pass

const _FOO: fn() -> String = || "foo".into();

pub fn bar() -> fn() -> String {
    || "bar".into()
}

fn main(){}
