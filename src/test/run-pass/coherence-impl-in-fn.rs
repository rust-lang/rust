fn main() {
    enum x { foo }
    impl x : core::cmp::Eq {
        pure fn eq(&self, other: &x) -> bool {
            (*self) as int == (*other) as int
        }
        pure fn ne(&self, other: &x) -> bool { !(*self).eq(other) }
    }
}
