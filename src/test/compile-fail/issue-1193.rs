// error-pattern: pattern variable conflicts
mod foo {
    type t = u8;

    const a : t = 0u8;
    const b : t = 1u8;

    fn bar(v: t) -> bool {
        alt v {
            a { return true; }
            b { return false; }
        }
    }
}
