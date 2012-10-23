fn f1<T: copy>() -> T { }
//~^ ERROR obsolete syntax: lower-case kind bounds

fn f1<T: send>() -> T { }
//~^ ERROR obsolete syntax: lower-case kind bounds

fn f1<T: const>() -> T { }
//~^ ERROR obsolete syntax: lower-case kind bounds

fn f1<T: owned>() -> T { }
//~^ ERROR obsolete syntax: lower-case kind bounds

struct s {
    let foo: (),
    //~^ ERROR obsolete syntax: `let` in field declaration
    bar: ();
    //~^ ERROR obsolete syntax: field declaration terminated with semicolon
    new() { }
    //~^ ERROR obsolete syntax: struct constructor
}

struct ss {
    fn foo() { }
    //~^ ERROR obsolete syntax: class method
    #[whatever]
    fn foo() { }
    //~^ ERROR obsolete syntax: class method
}

struct q : r {
    //~^ ERROR obsolete syntax: class traits
}

struct sss {
    priv {
    //~^ ERROR obsolete syntax: private section
        foo: ()
    }
}

fn obsolete_with() {
    struct S {
        foo: (),
        bar: (),
    }

    let a = S { foo: (), bar: () };
    let b = S { foo: () with a };
    //~^ ERROR obsolete syntax: with
    let c = S { foo: (), with a };
    //~^ ERROR obsolete syntax: with
    let a = { foo: (), bar: () };
    let b = { foo: () with a };
    //~^ ERROR obsolete syntax: with
    let c = { foo: (), with a };
    //~^ ERROR obsolete syntax: with
}

fn obsolete_fixed_length_vec() {
    let foo: [int]/1;
    //~^ ERROR obsolete syntax: fixed-length vector
    foo = [1]/_;
    //~^ ERROR obsolete syntax: fixed-length vector
    let foo: [int]/1;
    //~^ ERROR obsolete syntax: fixed-length vector
    foo = [1]/1;
    //~^ ERROR obsolete syntax: fixed-length vector
}

fn obsolete_moves() {
    let mut x = 0;
    let y <- x;
    //~^ ERROR obsolete syntax: initializer-by-move
    y <- x; 
    //~^ ERROR obsolete syntax: binary move
}

fn main() { }
