fn f() {}
fn g<'a>() -> &'a u8 { loop {} }

fn main() {
    f::<'static>(); //~ ERROR E0088
    g::<'static, 'static>(); //~ ERROR E0088
}
