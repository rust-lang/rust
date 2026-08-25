fn with<F>(f: F) where F: FnOnce(&String) {}

fn arg_item(&_x: &String) {}
    //~^ ERROR [E0507]

fn arg_closure() {
    with(|&_x| ())
    //~^ ERROR [E0507]
}

fn let_pat() {
    let &_x = &"hi".to_string();
    //~^ ERROR [E0507]
}

pub fn main() {}
