fn main() {
    #[allow(warnings)] if true {} //~ ERROR attributes are not yet allowed on `if` expressions
    if false {
    } else if true {
    }

	#[allow(warnings)] if let Some(_) = Some(true) { //~ ERROR attributes are not yet allowed on `if` expressions
	}
}
