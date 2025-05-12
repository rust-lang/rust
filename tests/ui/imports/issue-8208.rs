use self::*; //~ ERROR: unresolved import `self::*` [E0432]
             //~^ NOTE cannot glob-import a module into itself

mod foo {
    use foo::*; //~ ERROR: unresolved import `foo::*` [E0432]
                //~^ NOTE cannot glob-import a module into itself

    mod bar {
        use super::bar::*;
        //~^ ERROR: unresolved import `super::bar::*` [E0432]
        //~| NOTE cannot glob-import a module into itself
    }

}

fn main() {
}
