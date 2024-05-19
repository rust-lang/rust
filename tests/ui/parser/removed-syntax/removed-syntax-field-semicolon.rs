struct S {
    bar: ();
    //~^ ERROR struct fields are separated by `,`
}

fn main() {}
