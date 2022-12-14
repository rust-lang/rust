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
    //~^ ERROR no method named `x` found
    func.x();
    //~^ ERROR no method named `x` found
}
