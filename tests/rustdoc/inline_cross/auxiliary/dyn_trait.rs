pub type Ty0 = dyn for<'any> FnOnce(&'any str) -> bool;

pub type Ty1<'obj> = dyn std::fmt::Display + 'obj;

pub type Ty2 = dyn for<'a, 'r> Container<'r, Item<'a, 'static> = ()>;

pub type Ty3<'s> = &'s dyn ToString;

pub fn func0(_: &(dyn Fn() + '_)) {}

pub fn func1<'func>(_: &(dyn Fn() + 'func)) {}

pub trait Container<'r> {
    type Item<'a, 'ctx>;
}

pub trait Shape<'a> {}
