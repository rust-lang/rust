pub fn foo(x: &u32) -> u32 {
    /* Safety:
     * This is totally ok.
     */
    unsafe { *(x as *const u32) }
}
