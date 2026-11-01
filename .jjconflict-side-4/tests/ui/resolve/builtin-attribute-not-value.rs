fn missing_value() {
    path;
    //~^ ERROR cannot find value `path` in this scope [E0423]
}

fn missing_function() {
    path();
    //~^ ERROR cannot find function `path` in this scope [E0423]
}

fn suggested_value() {
    let pathh = ();
    path;
    //~^ ERROR cannot find value `path` in this scope [E0423]
    let _ = pathh;
}

struct Wat {
    path: (),
}

impl Wat {
    fn new() -> Wat {
        Wat { path }
        //~^ ERROR cannot find value `path` in this scope [E0423]
    }
}

fn main() {}
