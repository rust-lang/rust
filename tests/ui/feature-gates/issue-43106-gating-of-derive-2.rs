// This test checks cases where the derive-macro does not exist.

mod derive {
    #[derive(x3300)]
    //~^ ERROR cannot find derive macro `x3300` in this scope
    //~| ERROR cannot find derive macro `x3300` in this scope
    union U { f: i32 }

    #[derive(x3300)]
    //~^ ERROR cannot find derive macro `x3300` in this scope
    //~| ERROR cannot find derive macro `x3300` in this scope
    enum E { }

    #[derive(x3300)]
    //~^ ERROR cannot find derive macro `x3300` in this scope
    //~| ERROR cannot find derive macro `x3300` in this scope
    struct S;
}

fn main() {}
