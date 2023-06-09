#![crate_type = "rlib"]
#![feature(generic_arg_infer)]

struct Foo<const N: usize>;
struct Bar<T, const N: usize>(T);

fn arr_fn() -> [u8; _] {
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
    [0; 3]
}

fn ty_fn() -> Bar<i32, _> {
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
    Bar::<i32, 3>(0)
}

fn ty_fn_mixed() -> Bar<_, _> {
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
    Bar::<i32, 3>(0)
}

const ARR_CT: [u8; _] = [0; 3];
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for constants
static ARR_STATIC: [u8; _] = [0; 3];
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for static variables
const TY_CT: Bar<i32, _> = Bar::<i32, 3>(0);
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for constants
static TY_STATIC: Bar<i32, _> = Bar::<i32, 3>(0);
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for static variables
const TY_CT_MIXED: Bar<_, _> = Bar::<i32, 3>(0);
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for constants
static TY_STATIC_MIXED: Bar<_, _> = Bar::<i32, 3>(0);
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for static variables
trait ArrAssocConst {
    const ARR: [u8; _];
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for constants
}
trait TyAssocConst {
    const ARR: Bar<i32, _>;
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for constants
}
trait TyAssocConstMixed {
    const ARR: Bar<_, _>;
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for constants
}

trait AssocTy {
    type Assoc;
}
impl AssocTy for i8 {
    type Assoc = [u8; _];
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for associated types
}
impl AssocTy for i16 {
    type Assoc = Bar<i32, _>;
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for associated types
}
impl AssocTy for i32 {
    type Assoc = Bar<_, _>;
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for associated types
}
