enum no0 {
    x0(&uint) //~ ERROR to use region types here, the containing type must be declared with a region bound
}

enum no1 {
    x1(&self.uint) //~ ERROR to use region types here, the containing type must be declared with a region bound
}

enum no2 {
    x2(&foo.uint) //~ ERROR named regions other than `self` are not allowed as part of a type declaration
}

enum yes0/& {
    x3(&uint)
}

enum yes1/& {
    x4(&self.uint)
}

enum yes2/& {
    x5(&foo.uint) //~ ERROR named regions other than `self` are not allowed as part of a type declaration
}

fn main() {}