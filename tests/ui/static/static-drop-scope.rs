struct WithDtor;

impl Drop for WithDtor {
    fn drop(&mut self) {}
}

static EARLY_DROP_S: i32 = (WithDtor, 0).1;
//~^ ERROR destructor of

const EARLY_DROP_C: i32 = (WithDtor, 0).1;
//~^ ERROR destructor of

const fn const_drop<T>(_: T) {}
//~^ ERROR destructor of

const fn const_drop2<T>(x: T) {
    (x, ()).1
    //~^ ERROR destructor of
}

const EARLY_DROP_C_OPTION: i32 = (Some(WithDtor), 0).1;
//~^ ERROR destructor of

const HELPER: Option<WithDtor> = Some(WithDtor);

const EARLY_DROP_C_OPTION_CONSTANT: i32 = (HELPER, 0).1;
//~^ ERROR destructor of

fn main () {}
