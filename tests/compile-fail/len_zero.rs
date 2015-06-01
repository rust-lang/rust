#![feature(plugin)]
#![plugin(clippy)]

struct One;

#[deny(len_without_is_empty)]
impl One {
	fn len(self: &Self) -> isize { //~ERROR Item 'One' has a '.len()' method
		1
	}
}

#[deny(len_without_is_empty)]
trait TraitsToo {
	fn len(self: &Self) -> isize; //~ERROR Trait 'TraitsToo' has a '.len()' method,
}

impl TraitsToo for One {
	fn len(self: &Self) -> isize {
		0
	}
}

#[allow(dead_code)]
struct HasIsEmpty;

#[deny(len_without_is_empty)]
#[allow(dead_code)]
impl HasIsEmpty {
	fn len(self: &Self) -> isize {
		1
	}
	
	fn is_empty() -> bool {
		false
	}
}

#[deny(len_zero)]
fn main() {
	let x = [1, 2];
	if x.len() == 0 { //~ERROR Consider replacing the len comparison
		println!("This should not happen!");
	}
	
	let y = One;
	if y.len()  == 0 { //no error because One does not have .is_empty()
		println!("This should not happen either!");
	}
	
	let z : &TraitsToo = &y;
	if z.len() > 0 { //~ERROR Consider replacing the len comparison
		println!("Nor should this!");
	}
}
