// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

use std::rc::Rc;

pub fn main() {
    let _x = Rc::new(vec![1, 2]).into_iter();
    //[ast]~^ ERROR cannot move out of borrowed content [E0507]
    //[mir]~^^ ERROR [E0507]
}
