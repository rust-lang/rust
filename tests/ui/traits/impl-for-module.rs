mod a {
}

trait A {
}

impl A for a { //~ ERROR expected type, found module
}

fn main() {
}
