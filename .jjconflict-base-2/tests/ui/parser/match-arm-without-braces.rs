struct S;

impl S {
    fn get<K, V: Default>(_: K) -> Option<V> {
        Default::default()
    }
}

enum Val {
    Foo,
    Bar,
}

impl Default for Val {
    fn default() -> Self {
        Val::Foo
    }
}

fn main() {
    match S::get(1) {
        Some(Val::Foo) => {}
        _ => {}
    }
    match S::get(2) {
        Some(Val::Foo) => 3; //~ ERROR `match` arm body without braces
        _ => 4,
    }
    match S::get(5) {
        Some(Val::Foo) =>
          7; //~ ERROR `match` arm body without braces
          8;
        _ => 9,
    }
    match S::get(10) {
        Some(Val::Foo) =>
          11; //~ ERROR `match` arm body without braces
          12;
        _ => (),
    }
    match S::get(13) {
        None => {}
        Some(Val::Foo) =>
          14; //~ ERROR `match` arm body without braces
          15;
    }
    match S::get(16) {
        Some(Val::Foo) => 17 //~ ERROR expected `,` following `match` arm
        _ => 18,
    };
    match S::get(19) {
        Some(Val::Foo) =>
          20; //~ ERROR `match` arm body without braces
          21
        _ => 22,
    }
    match S::get(23) {
        Some(Val::Foo) =>
          24; //~ ERROR `match` arm body without braces
          25
        _ => (),
    }
    match S::get(26) {
        None => {}
        Some(Val::Foo) =>
          27; //~ ERROR `match` arm body without braces
          28
    }
    match S::get(29) {
        Some(Val::Foo) =>
          30; //~ ERROR expected one of
          31,
        _ => 32,
    }
    match S::get(33) {
        Some(Val::Foo) =>
          34; //~ ERROR expected one of
          35,
        _ => (),
    }
    match S::get(36) {
        None => {}
        Some(Val::Foo) =>
          37; //~ ERROR expected one of
          38,
    }
}
