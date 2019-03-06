pub trait ThisTrait {
    fn asdf(&self);

    /// let's link to [`asdf`](ThisTrait::asdf)
    fn qwop(&self);
}
