#![feature(plugin)]
#![plugin(clippy)]

struct One;

#[deny(len_without_is_empty)]
impl One {
    fn len(self: &Self) -> isize { //~ERROR item `One` has a `.len(_: &Self)`
        1
    }
}

#[deny(len_without_is_empty)]
trait TraitsToo {
    fn len(self: &Self) -> isize; //~ERROR trait `TraitsToo` has a `.len(_:
}

impl TraitsToo for One {
    fn len(self: &Self) -> isize {
        0
    }
}

struct HasIsEmpty;

#[deny(len_without_is_empty)]
impl HasIsEmpty {
    fn len(self: &Self) -> isize {
        1
    }

    fn is_empty(self: &Self) -> bool {
        false
    }
}

struct Wither;

#[deny(len_without_is_empty)]
trait WithIsEmpty {
    fn len(self: &Self) -> isize;
    fn is_empty(self: &Self) -> bool;
}

impl WithIsEmpty for Wither {
    fn len(self: &Self) -> isize {
        1
    }

    fn is_empty(self: &Self) -> bool {
        false
    }
}

struct HasWrongIsEmpty;

#[deny(len_without_is_empty)]
impl HasWrongIsEmpty {
    fn len(self: &Self) -> isize { //~ERROR item `HasWrongIsEmpty` has a `.len(_: &Self)`
        1
    }

    #[allow(dead_code, unused)]
    fn is_empty(self: &Self, x : u32) -> bool {
        false
    }
}

#[deny(len_zero)]
fn main() {
    let x = [1, 2];
    if x.len() == 0 { //~ERROR consider replacing the len comparison
        println!("This should not happen!");
    }

    let y = One;
    if y.len()  == 0 { //no error because One does not have .is_empty()
        println!("This should not happen either!");
    }

    let z : &TraitsToo = &y;
    if z.len() > 0 { //no error, because TraitsToo has no .is_empty() method
        println!("Nor should this!");
    }

    let hie = HasIsEmpty;
    if hie.len() == 0 { //~ERROR consider replacing the len comparison
        println!("Or this!");
    }
    if hie.len() != 0 { //~ERROR consider replacing the len comparison
        println!("Or this!");
    }
    if hie.len() > 0 { //~ERROR consider replacing the len comparison
        println!("Or this!");
    }
    assert!(!hie.is_empty());

    let wie : &WithIsEmpty = &Wither;
    if wie.len() == 0 { //~ERROR consider replacing the len comparison
        println!("Or this!");
    }
    assert!(!wie.is_empty());

    let hwie = HasWrongIsEmpty;
    if hwie.len() == 0 { //no error as HasWrongIsEmpty does not have .is_empty()
        println!("Or this!");
    }
}
