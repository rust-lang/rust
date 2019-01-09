mod stuff {
    pub struct Item {
        c_object: Box<CObj>,
    }
    pub struct CObj {
        name: Option<String>,
    }
    impl Item {
        pub fn new() -> Item {
            Item {
                c_object: Box::new(CObj { name: None }),
            }
        }
    }
}

macro_rules! check_ptr_exist {
    ($var:expr, $member:ident) => (
        (*$var.c_object).$member.is_some()
        //~^ ERROR field `name` of struct `stuff::CObj` is private
        //~^^ ERROR field `c_object` of struct `stuff::Item` is private
    );
}

fn main() {
    let item = stuff::Item::new();
    println!("{}", check_ptr_exist!(item, name));
}
