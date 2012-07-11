type item_ty_yes0 = {
    x: &uint
};

type item_ty_yes1 = {
    x: &self.uint
};

type item_ty_yes2 = {
    x: &foo.uint //~ ERROR named regions other than `self` are not allowed as part of a type declaration
};

fn main() {}