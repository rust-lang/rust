// Here, `'a` and `'c` are late-bound and `'b`, `'d`, `T` and `N` are early-bound.

pub fn f<'a, 'b, 'c, 'd, T, const N: usize>(_: impl Copy)
where
    'b:,
    'd:,
{}

pub struct Ty;

impl Ty {
    pub fn f<'a, 'b, 'c, 'd, T, const N: usize>(_: impl Copy)
    where
        'b:,
        'd:,
    {}
}
