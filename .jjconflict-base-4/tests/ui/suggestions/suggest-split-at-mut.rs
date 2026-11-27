fn foo() {
    let mut foo = [1, 2, 3, 4];
    let a = &mut foo[2];
    let b = &mut foo[3]; //~ ERROR cannot borrow `foo[_]` as mutable more than once at a time
    *a = 5;
    *b = 6;
    println!("{:?} {:?}", a, b);
}

fn bar() {
    let mut foo = [1,2,3,4];
    let a = &mut foo[..2];
    let b = &mut foo[2..]; //~ ERROR cannot borrow `foo` as mutable more than once at a time
    a[0] = 5;
    b[0] = 6;
    println!("{:?} {:?}", a, b);
}

fn baz() {
    let mut foo = [1,2,3,4];
    let a = &foo[..2];
    let b = &mut foo[2..]; //~ ERROR cannot borrow `foo` as mutable because it is also borrowed as immutable
    b[0] = 6;
    println!("{:?} {:?}", a, b);
}

fn qux() {
    let mut foo = [1,2,3,4];
    let a = &mut foo[..2];
    let b = &foo[2..]; //~ ERROR cannot borrow `foo` as immutable because it is also borrowed as mutable
    a[0] = 5;
    println!("{:?} {:?}", a, b);
}

fn bad() {
    let mut foo = [1,2,3,4];
    let a = &foo[1];
    let b = &mut foo[2]; //~ ERROR cannot borrow `foo[_]` as mutable because it is also borrowed as immutable
    *b = 6;
    println!("{:?} {:?}", a, b);
}

fn bat() {
    let mut foo = [1,2,3,4];
    let a = &mut foo[1];
    let b = &foo[2]; //~ ERROR cannot borrow `foo[_]` as immutable because it is also borrowed as mutable
    *a = 5;
    println!("{:?} {:?}", a, b);
}

fn ang() {
    let mut foo = [1,2,3,4];
    let a = &mut foo[0..];
    let b = &foo[0..]; //~ ERROR cannot borrow `foo` as immutable because it is also borrowed as mutable
    a[0] = 5;
    println!("{:?} {:?}", a, b);
}

fn main() {
    foo();
    bar();
}
