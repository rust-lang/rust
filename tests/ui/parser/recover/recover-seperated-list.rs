macro_rules! probe {
    ($ty:ty) => {
        compile_error!("TY FRAGMENT MATCHED")
    };
}

fn f<'a, 'b>() -> probe!(impl Sized + use<'a 'b>) {
    //~^ ERROR expected one of `,` or `>`, found `'b`
    loop {}
}

fn main() {

}
