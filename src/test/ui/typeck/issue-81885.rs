const TEST4: fn() -> _ = 42;
                  //~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions

fn main() {
    const TEST5: fn() -> _ = 42;
                      //~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions

}
