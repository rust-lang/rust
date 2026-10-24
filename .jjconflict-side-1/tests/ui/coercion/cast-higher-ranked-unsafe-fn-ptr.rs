//@ check-pass

fn higher_ranked_fndef(ctx: &mut ()) {}

fn test(higher_ranked_fnptr: fn(&mut ())) {
    fn as_unsafe<T>(_: unsafe fn(T)) {}

    // Make sure that we can cast higher-ranked fn items and pointers to
    // a non-higher-ranked target.
    as_unsafe(higher_ranked_fndef);
    as_unsafe(higher_ranked_fnptr);
}

fn main() {}
