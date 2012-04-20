resource no0(x: &uint) { //! ERROR to use region types here, the containing type must be declared with a region bound
}

resource no1(x: &self.uint) { //! ERROR to use region types here, the containing type must be declared with a region bound
}

resource no2(x: &foo.uint) { //! ERROR named regions other than `self` are not allowed as part of a type declaration
}

resource yes0/&(x: &uint) {
}

resource yes1/&(x: &self.uint) {
}

resource yes2/&(x: &foo.uint) { //! ERROR named regions other than `self` are not allowed as part of a type declaration
}

fn main() {}