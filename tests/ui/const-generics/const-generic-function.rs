fn foo<const N: i32>() -> i32 {
    N
}

const fn bar(n: i32, m: i32) -> i32 {
    n
}

const fn baz() -> i32 {
    1
}

const FOO: i32 = 3;

fn main() {
    foo::<baz()>(); //~ ERROR expected type, found function `baz`
    //~| ERROR unresolved item provided when a constant was expected
    foo::<bar(bar(1, 1), bar(1, 1))>(); //~ ERROR expected type, found `1`
    foo::<bar(1, 1)>(); //~ ERROR expected type, found `1`
    foo::<bar(FOO, 2)>(); //~ ERROR expected type, found `2`
}
