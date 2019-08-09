// check that borrowck looks inside consts/statics

static FN : &'static (dyn Fn() -> (Box<dyn Fn()->Box<i32>>) + Sync) = &|| {
    let x = Box::new(0);
    Box::new(|| x) //~ ERROR cannot move out of `x`, a captured variable in an `Fn` closure
};

fn main() {
    let f = (FN)();
    f();
    f();
}
