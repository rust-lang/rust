struct ret;
struct obj;

impl obj {
    fn func() -> ret {
        ret
    }
}

fn func() -> ret {
    ret
}

fn main() {
    obj::func.x();
    //~^ ERROR no method named `x` found for type `fn() -> ret {obj::func}` in the current scope
    func.x();
    //~^ ERROR no method named `x` found for type `fn() -> ret {func}` in the current scope
}
