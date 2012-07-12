enum yes0 {
    x3(&uint)
}

enum yes1 {
    x4(&self/uint)
}

enum yes2 {
    x5(&foo/uint) //~ ERROR named regions other than `self` are not allowed as part of a type declaration
}

fn main() {}