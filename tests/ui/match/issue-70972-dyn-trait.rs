const F: &'static dyn Send = &7u32;

fn main() {
    let a: &dyn Send = &7u32;
    match a {
        F => panic!(),
        //~^ ERROR `dyn Send` cannot be used in patterns
        _ => {}
    }
}
