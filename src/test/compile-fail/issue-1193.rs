// error-pattern: pattern variable conflicts
mod foo {
    #[legacy_exports];
    type t = u8;

    const a : t = 0u8;
    const b : t = 1u8;

    fn bar(v: t) -> bool {
        match v {
            a => { return true; }
            b => { return false; }
        }
    }
}
