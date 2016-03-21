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
    if x.len() == 0 {
        //~^ERROR length comparison to zero
        //~|HELP consider using `is_empty`
        //~|SUGGESTION x.is_empty()
        println!("This should not happen!");
    }

    if "".len() == 0 {
        //~^ERROR length comparison to zero
        //~|HELP consider using `is_empty`
        //~|SUGGESTION "".is_empty()
    }

    let y = One;
    if y.len()  == 0 { //no error because One does not have .is_empty()
        println!("This should not happen either!");
    }

    let z : &TraitsToo = &y;
    if z.len() > 0 { //no error, because TraitsToo has no .is_empty() method
        println!("Nor should this!");
    }

    let has_is_empty = HasIsEmpty;
    if has_is_empty.len() == 0 {
        //~^ERROR length comparison to zero
        //~|HELP consider using `is_empty`
        //~|SUGGESTION has_is_empty.is_empty()
        println!("Or this!");
    }
    if has_is_empty.len() != 0 {
        //~^ERROR length comparison to zero
        //~|HELP consider using `is_empty`
        //~|SUGGESTION !has_is_empty.is_empty()
        println!("Or this!");
    }
    if has_is_empty.len() > 0 {
        //~^ERROR length comparison to zero
        //~|HELP consider using `is_empty`
        //~|SUGGESTION !has_is_empty.is_empty()
        println!("Or this!");
    }
    assert!(!has_is_empty.is_empty());

    let with_is_empty: &WithIsEmpty = &Wither;
    if with_is_empty.len() == 0 {
        //~^ERROR length comparison to zero
        //~|HELP consider using `is_empty`
        //~|SUGGESTION with_is_empty.is_empty()
        println!("Or this!");
    }
    assert!(!with_is_empty.is_empty());

    let has_wrong_is_empty = HasWrongIsEmpty;
    if has_wrong_is_empty.len() == 0 { //no error as HasWrongIsEmpty does not have .is_empty()
        println!("Or this!");
    }
}
