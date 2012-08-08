trait me {
    fn me() -> uint;
}
impl uint: me { fn me() -> uint { self } }
