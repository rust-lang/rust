// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

fn with<F>(f: F) where F: FnOnce(&String) {}

fn arg_item(&_x: &String) {}
    //[ast]~^ ERROR cannot move out of borrowed content [E0507]
    //[mir]~^^ ERROR [E0507]

fn arg_closure() {
    with(|&_x| ())
    //[ast]~^ ERROR cannot move out of borrowed content [E0507]
    //[mir]~^^ ERROR [E0507]
}

fn let_pat() {
    let &_x = &"hi".to_string();
    //[ast]~^ ERROR cannot move out of borrowed content [E0507]
    //[mir]~^^ ERROR [E0507]
}

pub fn main() {}
