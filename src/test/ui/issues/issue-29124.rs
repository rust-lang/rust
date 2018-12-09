struct Ret;
struct Obj;

impl Obj {
    fn func() -> Ret {
        Ret
    }
}

fn func() -> Ret {
    Ret
}

fn main() {
    Obj::func.x();
    //~^ ERROR no method named `x` found for type `fn() -> Ret {<Obj>::func}` in the current scope
    func.x();
    //~^ ERROR no method named `x` found for type `fn() -> Ret {func}` in the current scope
}
