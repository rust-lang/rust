// This test checks cases where the derive-macro does not exist.

mod derive {
    #[derive(x3300)]
    //~^ ERROR cannot find derive macro `x3300`
    //~| ERROR cannot find derive macro `x3300`
    union U { f: i32 }

    #[derive(x3300)]
    //~^ ERROR cannot find derive macro `x3300`
    //~| ERROR cannot find derive macro `x3300`
    enum E { }

    #[derive(x3300)]
    //~^ ERROR cannot find derive macro `x3300`
    //~| ERROR cannot find derive macro `x3300`
    struct S;
}

fn main() {}
