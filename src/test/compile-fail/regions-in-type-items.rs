type item_ty_no0 = {
    x: &uint //! ERROR to use region types here, the containing type must be declared with a region bound
};

type item_ty_no1 = {
    x: &self.uint //! ERROR to use region types here, the containing type must be declared with a region bound
};

type item_ty_no2 = {
    x: &foo.uint //! ERROR named regions other than `self` are not allowed as part of a type declaration
};

type item_ty_yes0/& = {
    x: &uint
};

type item_ty_yes1/& = {
    x: &self.uint
};

type item_ty_yes2/& = {
    x: &foo.uint //! ERROR named regions other than `self` are not allowed as part of a type declaration
};

fn main() {}