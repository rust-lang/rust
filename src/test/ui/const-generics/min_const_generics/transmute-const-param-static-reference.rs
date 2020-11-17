struct Const<const P: &'static ()>;
//~^ ERROR `&'static ()` is forbidden as the type of a const generic parameter

fn main() {
    const A: &'static () = unsafe {
        std::mem::transmute(10 as *const ())
    };

    let _ = Const::<{A}>;
}
