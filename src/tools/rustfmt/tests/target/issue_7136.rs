// Original issue
macro_rules! example {
    ($func: ident <>) => {{
        $func();
    }};
}

fn log() {
    println!("Hi");
}

fn main() {
    example!(log<>);
}

// Case 1
macro_rules! m {
    ($f:ident <>) => {
        $f();
    };
}
fn log() {}
fn main() {
    m!(log<>);
}

// Case 2
macro_rules! m {
    ($f:ident :: <>) => {
        $f();
    };
}
fn log() {}
fn main() {
    m!(log::<>);
}

// Case 3
macro_rules! m {
    ($f:ident :: <> ($($a:tt)*)) => {};
}
fn main() {
    m!(S::<>());
}

// Case 4
macro_rules! m {
    ($m:ident :: $n:ident <>) => {};
}
fn main() {
    m!(N::Foo<>);
}

// Case 5
macro_rules! m {
    ($f:ident <>) => {};
}
fn main() {
    m!(Foo<>);
}

// Case 6
macro_rules! m {
    ($a:ident <>, $b:ident <>) => {};
}
fn main() {
    m!(log<>, foo<>);
}

// Case 7
macro_rules! m {
    ($a:ident as $t:ident <>) => {};
}
fn main() {
    m!(a as Foo<>);
}
